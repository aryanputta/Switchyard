// Switchyard online search API.
//
// This is the serving path. Python owns research and trains the route
// classifier; this service loads the frozen model artifact (results/router_model.json)
// and applies the exact same feature extraction and linear decision rule, so the
// deployed router and the offline-evaluated router are identical by construction.
//
// The retrieval routes themselves (BM25 and HNSW over OpenSearch, Redis cache,
// cross-encoder rerank) attach where marked. The routing decision, the SLO
// deadline handling, the fallback policy, and Prometheus metrics are real here.
//
// Stdlib only, so it builds with no module downloads.
package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"
)

type routerModel struct {
	Routes  []string    `json:"routes"`
	Weights [][]float64 `json:"weights"`
}

var featureOrder = []string{
	"token_count", "char_count", "has_question", "has_model_number",
	"has_money", "has_negation", "has_compatibility", "has_color",
}

var (
	tokenRe   = regexp.MustCompile(`[a-z0-9]+(?:'[a-z0-9]+)?`)
	modelRe   = regexp.MustCompile(`\b(?:[a-z0-9-]*[a-z][a-z0-9-]*\d|[a-z0-9-]*\d[a-z0-9-]*[a-z])[a-z0-9-]*\b`)
	moneyRe   = regexp.MustCompile(`\$\s?\d+(?:\.\d+)?|\b\d+\s?(?:dollars|usd)\b`)
	negationR = regexp.MustCompile(`\b(without|no|not|free of|excluding|except|minus)\b`)
	questionR = regexp.MustCompile(`\b(how|what|why|when|where|who|which|does|is|are|can)\b`)
	compatR   = regexp.MustCompile(`\b(for|compatible|fits|works with|replacement for)\b`)
)

var colors = map[string]bool{
	"black": true, "white": true, "red": true, "blue": true, "green": true,
	"silver": true, "gray": true, "grey": true, "gold": true, "pink": true,
	"purple": true, "yellow": true, "brown": true, "beige": true, "rose": true,
}

// modeled route latencies, mirroring routes.ROUTE_COST_MS until measured under load
var routeBudgetMs = map[string]float64{
	"lexical": 2, "dense": 8, "hybrid": 11, "rerank": 35,
}

func featureVector(query string) []float64 {
	lower := strings.ToLower(query)
	tokens := tokenRe.FindAllString(lower, -1)
	b := func(v bool) float64 {
		if v {
			return 1
		}
		return 0
	}
	hasColor := false
	for _, t := range tokens {
		if colors[t] {
			hasColor = true
			break
		}
	}
	feats := map[string]float64{
		"token_count":       float64(len(tokens)) / 10.0,
		"char_count":        float64(len(query)) / 60.0,
		"has_question":      b(strings.Contains(query, "?") || questionR.MatchString(lower)),
		"has_model_number":  b(hasModelNumber(lower)),
		"has_money":         b(moneyRe.MatchString(lower)),
		"has_negation":      b(negationR.MatchString(lower)),
		"has_compatibility": b(compatR.MatchString(lower)),
		"has_color":         b(hasColor),
	}
	x := []float64{1.0}
	for _, name := range featureOrder {
		x = append(x, feats[name])
	}
	return x
}

// a token with both a letter and a digit, length >= 4
func hasModelNumber(lower string) bool {
	for _, tok := range tokenRe.FindAllString(lower, -1) {
		if len(tok) < 4 {
			continue
		}
		hasL, hasD := false, false
		for _, r := range tok {
			if r >= 'a' && r <= 'z' {
				hasL = true
			}
			if r >= '0' && r <= '9' {
				hasD = true
			}
		}
		if hasL && hasD {
			return true
		}
	}
	return false
}

func (m *routerModel) predict(query string) string {
	x := featureVector(query)
	best, bestScore := m.Routes[0], math.Inf(-1)
	for i, row := range m.Weights {
		s := 0.0
		for j, w := range row {
			s += w * x[j]
		}
		if s > bestScore {
			bestScore, best = s, m.Routes[i]
		}
	}
	return best
}

type server struct {
	model *routerModel
	mu    sync.Mutex
	count map[string]int64
}

type searchRequest struct {
	Query      string  `json:"query"`
	DeadlineMs float64 `json:"deadline_ms,omitempty"`
}

type searchResponse struct {
	Query     string  `json:"query"`
	Route     string  `json:"route"`
	BudgetMs  float64 `json:"route_budget_ms"`
	Downgrade bool    `json:"downgraded_for_deadline"`
	Note      string  `json:"note"`
}

// cheaperRoute returns the most capable route that fits the deadline.
func cheaperRoute(chosen string, deadlineMs float64) (string, bool) {
	order := []string{"rerank", "hybrid", "dense", "lexical"}
	if deadlineMs <= 0 {
		return chosen, false
	}
	if routeBudgetMs[chosen] <= deadlineMs {
		return chosen, false
	}
	for _, r := range order {
		if routeBudgetMs[r] <= deadlineMs {
			return r, true
		}
	}
	return "lexical", true
}

func (s *server) handleSearch(w http.ResponseWriter, r *http.Request) {
	var req searchRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil || req.Query == "" {
		http.Error(w, "expected JSON body with non-empty 'query'", http.StatusBadRequest)
		return
	}
	route := s.model.predict(req.Query)
	route, down := cheaperRoute(route, req.DeadlineMs)

	s.mu.Lock()
	s.count[route]++
	s.mu.Unlock()

	note := "retrieval wiring (OpenSearch BM25/HNSW, Redis, rerank) attaches here"
	resp := searchResponse{
		Query:     req.Query,
		Route:     route,
		BudgetMs:  routeBudgetMs[route],
		Downgrade: down,
		Note:      note,
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	s.mu.Lock()
	defer s.mu.Unlock()
	fmt.Fprintln(w, "# HELP switchyard_route_selected_total Routes selected by the router.")
	fmt.Fprintln(w, "# TYPE switchyard_route_selected_total counter")
	for route, n := range s.count {
		fmt.Fprintf(w, "switchyard_route_selected_total{route=%q} %d\n", route, n)
	}
}

func main() {
	modelPath := getenv("SWITCHYARD_MODEL", "results/router_model.json")
	addr := getenv("SWITCHYARD_ADDR", ":8080")

	raw, err := os.ReadFile(modelPath)
	if err != nil {
		log.Fatalf("read model %s: %v", modelPath, err)
	}
	var model routerModel
	if err := json.Unmarshal(raw, &model); err != nil {
		log.Fatalf("parse model: %v", err)
	}
	if len(model.Routes) == 0 {
		log.Fatal("model has no routes")
	}

	s := &server{model: &model, count: map[string]int64{}}
	mux := http.NewServeMux()
	mux.HandleFunc("/search", s.handleSearch)
	mux.HandleFunc("/metrics", s.handleMetrics)
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, "ok")
	})

	srv := &http.Server{Addr: addr, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	log.Printf("switchyard serving on %s with %d routes", addr, len(model.Routes))
	log.Fatal(srv.ListenAndServe())
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
