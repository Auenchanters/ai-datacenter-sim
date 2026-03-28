# FastAPI server entry point.
# Exposes POST /tick endpoint: receives AI action payload, advances simulation, returns new state.
# Exposes GET /state endpoint: returns the current full game state JSON.
# Exposes GET /leaderboard endpoint: returns ranked scores across all running agents.
