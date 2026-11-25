export const config = {
  port: Number(process.env.API_PORT || 3000),
  engineUrl: process.env.ENGINE_URL || 'http://vectordb-engine:8080',
  databaseUrl: process.env.DATABASE_URL || 'postgresql://vectordb:vectordb@postgres:5432/vectordb',
  prefilterMaxRows: Number(process.env.PREFILTER_MAX_ROWS || 2500),
  prefilterMaxSelectivity: Number(process.env.PREFILTER_MAX_SELECTIVITY || 0.08),
  postfilterMaxExpand: Number(process.env.POSTFILTER_MAX_EXPAND || 20),
  engineMaxConnections: Number(process.env.ENGINE_MAX_CONNECTIONS || 32),
};
