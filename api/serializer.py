# State serializer.
# Converts internal game state (NumPy arrays, hardware objects) into token-efficient JSON payload.
# Only transmits thermal_hotspots exceeding safe temperature thresholds to conserve AI context tokens.
