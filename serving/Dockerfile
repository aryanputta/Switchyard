# Multi-stage build: compile the static Go binary, ship a minimal runtime image.
FROM golang:1.22-alpine AS build
WORKDIR /src
COPY serving/go.mod ./
COPY serving/main.go ./
RUN CGO_ENABLED=0 go build -o /out/switchyard .

FROM gcr.io/distroless/static-debian12
COPY --from=build /out/switchyard /switchyard
COPY results/router_model.json /results/router_model.json
ENV SWITCHYARD_MODEL=/results/router_model.json
ENV SWITCHYARD_ADDR=:8080
EXPOSE 8080
ENTRYPOINT ["/switchyard"]
