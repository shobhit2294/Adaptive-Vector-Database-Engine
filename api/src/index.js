import express from 'express';
import { config } from './config.js';
import { PostgresStore } from './db/postgres.js';
import { EngineClient } from './clients/engineClient.js';
import { HybridQueryOptimizer } from './optimizer/queryOptimizer.js';
import { createVectorRouter } from './routes/vectors.js';
import { httpRequestDuration, httpRequestsTotal, register } from './metrics/prometheus.js';

export function createApp({ pg, engine, optimizer }) {
  const app = express();
  app.use(express.json({ limit: '50mb' }));

  app.use((req, res, next) => {
    const start = performance.now();
    res.on('finish', () => {
      const route = req.route?.path || req.path;
      const secs = (performance.now() - start) / 1000;
      httpRequestsTotal.inc({ route, method: req.method, status: String(res.statusCode) });
      httpRequestDuration.observe({ route, method: req.method }, secs);
    });
    next();
  });

  app.get('/health', async (_req, res) => {
    try {
      const [pgOk, engHealth] = await Promise.all([pg.health(), engine.health()]);
      res.json({ status: 'ok', postgres: pgOk, engine: engHealth });
    } catch (err) {
      res.status(503).json({ status: 'degraded', error: err.message });
    }
  });

  app.get('/metrics', async (_req, res) => {
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  });

  app.use('/api/v1', createVectorRouter({ pg, engine, optimizer }));

  app.use((err, _req, res, _next) => {
    const status = err.status || 500;
    res.status(status).json({ error: err.message || 'internal error' });
  });

  return app;
}

async function main() {
  const pg = new PostgresStore(config.databaseUrl);
  await pg.initWithRetry();

  const engine = new EngineClient(config.engineUrl, config.engineMaxConnections);
  const optimizer = new HybridQueryOptimizer(pg, engine, {
    prefilterMaxRows: config.prefilterMaxRows,
    prefilterMaxSelectivity: config.prefilterMaxSelectivity,
    postfilterMaxExpand: config.postfilterMaxExpand,
  });

  const app = createApp({ pg, engine, optimizer });
  const server = app.listen(config.port, '0.0.0.0', () => {
    console.log(`Adaptive VectorDB Node.js API listening on 0.0.0.0:${config.port}`);
  });

  const shutdown = async () => {
    server.close(async () => {
      await engine.close().catch(() => {});
      await pg.close().catch(() => {});
      process.exit(0);
    });
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((err) => {
    console.error('Fatal startup error:', err);
    process.exit(1);
  });
}
