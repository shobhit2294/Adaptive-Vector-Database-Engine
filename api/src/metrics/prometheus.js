import client from 'prom-client';

export const register = new client.Registry();
client.collectDefaultMetrics({ register });

export const httpRequestsTotal = new client.Counter({
  name: 'api_http_requests_total',
  help: 'Total API HTTP requests',
  labelNames: ['route', 'method', 'status'],
  registers: [register],
});

export const httpRequestDuration = new client.Histogram({
  name: 'api_http_request_duration_seconds',
  help: 'API HTTP request duration in seconds',
  labelNames: ['route', 'method'],
  buckets: [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5],
  registers: [register],
});

export const optimizerPlanTotal = new client.Counter({
  name: 'api_optimizer_plan_total',
  help: 'Query optimizer execution plans selected',
  labelNames: ['plan'],
  registers: [register],
});

export const pgQueryDuration = new client.Histogram({
  name: 'api_postgres_query_duration_seconds',
  help: 'PostgreSQL metadata filtering/hydration latency in seconds',
  labelNames: ['operation'],
  buckets: [0.0005, 0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
  registers: [register],
});
