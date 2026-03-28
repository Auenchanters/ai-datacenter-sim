from typing import Dict, Any


HARDWARE_CATALOG: Dict[str, Dict[str, Any]] = {

    # -----------------------------------------------------------------------
    # SERVER RACKS
    # compute_units: max compute capacity when utilization = 1.0
    # power_idle_kw: power draw at utilization = 0.0
    # power_max_kw: power draw at utilization = 1.0
    # heat_coefficient: multiplier applied to power draw to derive heat output
    #                   (1.0 = all power becomes heat; liquid cooling < 1.0)
    # safe_temp_c: intake temperature above which throttling begins
    # critical_temp_c: intake temperature at which server shuts down
    # -----------------------------------------------------------------------

    "SERVER_CPU_BASIC": {
        "category": "server",
        "display_name": "Standard 1U CPU Rack",
        "cost": 10_000,
        "compute_units": 100,
        "power_idle_kw": 1.0,
        "power_max_kw": 5.0,
        "heat_coefficient": 1.0,
        "safe_temp_c": 27.0,
        "critical_temp_c": 45.0,
        "description": (
            "Low-cost, low-power rack. Best suited for web hosting and "
            "lightweight compute jobs. Easy to cool with basic airflow. "
            "Poor price-to-performance for heavy AI workloads."
        ),
    },

    "SERVER_GPU_H100": {
        "category": "server",
        "display_name": "High-Density AI GPU Rack (H100)",
        "cost": 50_000,
        "compute_units": 1_000,
        "power_idle_kw": 4.0,
        "power_max_kw": 20.0,
        "heat_coefficient": 1.0,
        "safe_temp_c": 25.0,
        "critical_temp_c": 40.0,
        "description": (
            "High-density GPU rack. Delivers maximum compute for AI training "
            "and inference jobs. Generates extreme exhaust heat. Requires "
            "dedicated cold aisle intake and hot aisle exhaust management. "
            "Will thermal throttle rapidly if placed without adequate cooling."
        ),
    },

    "SERVER_LIQUID_AI": {
        "category": "server",
        "display_name": "Direct-to-Chip Liquid Cooled AI Rack",
        "cost": 120_000,
        "compute_units": 2_500,
        "power_idle_kw": 5.0,
        "power_max_kw": 35.0,
        "heat_coefficient": 0.15,
        "safe_temp_c": 30.0,
        "critical_temp_c": 50.0,
        "description": (
            "Elite liquid-cooled rack. Heat is removed directly from chips "
            "via coolant loops, releasing only 15 percent of thermal energy "
            "into the room. Offers the highest compute density on the grid. "
            "Highest upfront cost. Dramatically reduces room cooling burden "
            "and improves PUE in hyper-dense deployments."
        ),
    },

    # -----------------------------------------------------------------------
    # COOLING UNITS
    # cooling_capacity_kw: maximum heat removed per tick at fan_speed = 1.0
    # power_max_kw: power consumed at fan_speed = 1.0
    # airflow_range: number of tiles forward that cold air reaches
    # -----------------------------------------------------------------------

    "COOLING_CRAC_BASIC": {
        "category": "cooling",
        "display_name": "Standard Perimeter CRAC Unit",
        "cost": 15_000,
        "cooling_capacity_kw": 30.0,
        "power_max_kw": 15.0,
        "airflow_range": 5,
        "description": (
            "Affordable perimeter air conditioning unit. Pushes cold air "
            "up to 5 tiles into the room. Low upfront cost but inefficient: "
            "uses 15 kW of electricity to remove 30 kW of heat (COP = 2.0). "
            "Suitable for low-density CPU server rows."
        ),
    },

    "COOLING_IN_ROW": {
        "category": "cooling",
        "display_name": "High-Efficiency In-Row Cooler",
        "cost": 30_000,
        "cooling_capacity_kw": 45.0,
        "power_max_kw": 10.0,
        "airflow_range": 2,
        "description": (
            "Precision in-row cooling unit placed directly beside server "
            "racks. Highly efficient: uses only 10 kW to remove 45 kW of heat "
            "(COP = 4.5). Short airflow range of 2 tiles requires careful "
            "placement directly adjacent to server intakes. Best for "
            "high-density GPU rows. Significantly improves PUE."
        ),
    },

    "COOLING_CHILLER_IND": {
        "category": "cooling",
        "display_name": "Industrial Floor Chiller",
        "cost": 80_000,
        "cooling_capacity_kw": 150.0,
        "power_max_kw": 40.0,
        "airflow_range": 5,
        "description": (
            "Large-scale industrial chiller. Floods a wide area with cold "
            "air. Designed for hyperscale deployments with many GPU racks. "
            "High upfront cost but best absolute cooling capacity available. "
            "Uses 40 kW to remove 150 kW of heat (COP = 3.75). Suitable "
            "for the AI to deploy once GPU density is high enough to justify "
            "the capital expenditure."
        ),
    },
}


def get_item(hardware_id: str) -> Dict[str, Any]:
    if hardware_id not in HARDWARE_CATALOG:
        raise KeyError(f"Hardware ID '{hardware_id}' not found in catalog.")
    return HARDWARE_CATALOG[hardware_id]


def get_servers() -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in HARDWARE_CATALOG.items() if v["category"] == "server"}


def get_cooling() -> Dict[str, Dict[str, Any]]:
    return {k: v for k, v in HARDWARE_CATALOG.items() if v["category"] == "cooling"}


def get_catalog_for_prompt() -> Dict[str, Any]:
    """
    Returns a minimal, token-efficient version of the catalog
    to inject into the AI system prompt.
    """
    result = {"servers": {}, "cooling": {}}
    for key, item in HARDWARE_CATALOG.items():
        entry = {
            "cost": item["cost"],
            "description": item["description"],
        }
        if item["category"] == "server":
            entry["compute_units"] = item["compute_units"]
            entry["power_idle_kw"] = item["power_idle_kw"]
            entry["power_max_kw"] = item["power_max_kw"]
            entry["safe_temp_c"] = item["safe_temp_c"]
            entry["critical_temp_c"] = item["critical_temp_c"]
            result["servers"][key] = entry
        else:
            entry["cooling_capacity_kw"] = item["cooling_capacity_kw"]
            entry["power_max_kw"] = item["power_max_kw"]
            entry["airflow_range"] = item["airflow_range"]
            result["cooling"][key] = entry
    return result
