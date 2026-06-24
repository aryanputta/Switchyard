// k6 load test for the Switchyard API. Mixed web and product traffic with a mix
// of tight and loose deadlines, so the router exercises both the cheap and the
// expensive routes and the deadline-downgrade path.
//
//   k6 run scripts/load_test.js
//
import http from "k6/http";
import { check } from "k6";

export const options = {
  scenarios: {
    steady: { executor: "constant-vus", vus: 20, duration: "30s" },
  },
  thresholds: {
    http_req_duration: ["p(95)<150", "p(99)<300"],
    http_req_failed: ["rate<0.01"],
  },
};

const queries = [
  { query: "sony wh-1000xm5" },
  { query: "comfortable shoes for nurses working long shifts" },
  { query: "black headphones under $100 without microphone", deadline_ms: 8 },
  { query: "replacement charger for dell xps 13 9310" },
  { query: "how does tail latency affect distributed search systems" },
  { query: "stainless steel water bottle 32 oz", deadline_ms: 4 },
];

const BASE = __ENV.SWITCHYARD_URL || "http://localhost:8080";

export default function () {
  const body = queries[Math.floor(Math.random() * queries.length)];
  const res = http.post(`${BASE}/search`, JSON.stringify(body), {
    headers: { "Content-Type": "application/json" },
  });
  check(res, {
    "status 200": (r) => r.status === 200,
    "has route": (r) => r.json("route") !== undefined,
  });
}
