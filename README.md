# ai-datacenter-sim
A headless 2D data center simulation environment designed to benchmark LLMs and RL agents on spatial reasoning, thermodynamics, and Power Usage Effectiveness (PUE) optimization


***

# AI Data Center Architect

## Project Vision and Overview

AI Data Center Architect is a management simulation game designed specifically as a benchmark for Large Language Models and Reinforcement Learning agents. The simulation evaluates AI agents by forcing them to act as the autonomous CEO and Lead Architect of a hyperscale data center. Agents must manage a budget, purchase physical hardware, process incoming compute workloads, and strategically place server racks on a 2D grid. The environment enforces complex real-world thermodynamics, including heat generation, directional airflow, and cooling efficiency. The ultimate objective is to determine which AI model achieves the highest profitability and the lowest Power Usage Effectiveness without causing thermal shutdowns.

## Core Game Mechanics

### Grid and Spatial Planning

The facility is a 2D matrix representing the physical floor plan. Every tile holds an ambient temperature value that updates per tick. The AI must place hardware coordinates on this grid. Servers utilize directional airflow, pulling cold air from the front intake and pushing hot air out the back exhaust. The AI must align servers to create dedicated hot aisles (exhausts facing each other) and cold aisles (intakes facing each other) to prevent thermal feedback loops.

### Thermodynamics and Heat Engine

Servers convert electrical power directly into heat based on their current workload utilization. Heat spreads across empty floor tiles using a Laplacian diffusion formula. The AI must purchase and place computer room air conditioning units to remove heat from the room. These cooling units consume significant electricity and must be oriented to push cold air into the server intakes. If a server's intake air exceeds its safe operating temperature, its performance throttles. If it reaches critical temperature, it shuts down and the assigned client contract fails.

### Economics and Workloads

The AI starts with a fixed capital expenditure budget to buy initial hardware. During the simulation, the AI pays operational expenditure for electricity consumed per tick. The engine generates dynamic workloads requiring specific compute units, offering a set financial reward per tick, and expiring after a set duration. The AI must assign these jobs to compatible servers to generate revenue.

### Primary Winning Metrics

Models are evaluated and ranked on a global leaderboard across three dimensions. Power Usage Effectiveness is the primary metric, calculated as total facility power divided by IT equipment power. A perfect score is 1.0, and any power wasted on over-cooling degrades this score. The second metric is total Return on Investment, representing the final bank balance after hardware and operational costs. The third metric is Service Level Agreement uptime, representing the percentage of workloads successfully completed without thermal throttling or hardware failure.

## Technology Stack

- Python 3.10+ for core engine logic and game loop management.
- NumPy for optimized 2D array matrix operations and heat diffusion calculations.
- FastAPI to handle tick progression, state generation, and parsing AI command payloads.
- LiteLLM to standardize API calls across OpenAI, Anthropic, Google, and local open-source models.
- Pygame for local 2D rendering and debugging during development.

## AI Interaction Architecture

### System Prompt and Context

At initialization, the AI receives a strict system prompt defining its persona, the physical rules of the simulation, and the hardware catalog. The catalog includes unit prices, maximum compute capacity, baseline power draw, and heat generation profiles for all available servers and cooling units.

### State Payload Input

The engine sends a JSON payload to the AI every tick. To conserve context window tokens, only critical data is transmitted.

- Global metrics including current tick, bank balance, current PUE, and total power draw.
- Thermal hotspots listing only grid coordinates that exceed safe temperature thresholds.
- Equipment arrays listing placed racks and cooling units, including coordinate data, facing direction, intake temperature, and current utilization.
- Workload market arrays listing available contracts and currently active jobs.

### Action Payload Output

The AI must respond with a strict JSON object containing its logic and discrete commands.

```json
{
  "thoughts": "Step-by-step logical deduction of the thermal state and financial strategy",
  "actions": [
    {"command": "BUY_EQUIPMENT", "type": "SERVER_GPU_H100", "position": {"x": 5, "y": 5}, "facing": "NORTH"},
    {"command": "ACCEPT_CONTRACT", "job_id": "job_002"}
  ]
}
```

## Development Roadmap

### Phase One Physics Engine

Build the headless 2D grid and mathematical rules without AI integration. Initialize the grid using a 2D NumPy array for base temperatures. Create object-oriented hardware classes for servers and cooling units with properties for position, facing direction, power draw, and heat output. Implement directional airflow logic to transfer heat from exhaust tiles. Implement Laplacian diffusion so heat naturally dissipates across empty tiles per tick. Build the core step function to calculate power costs, update temperatures, and advance time.

### Phase Two Economy Generator

Implement financial constraints and dynamic client jobs. Define the hardware catalog dictionary detailing costs and performance metrics. Create a workload spawner that generates contracts requiring compute units in exchange for currency. Build job assignment logic allowing servers to accept workloads, which dynamically increases their power draw and heat output based on utilization. Implement the billing cycle to calculate PUE and deduct combined electricity costs from the bank balance every tick.

### Phase Three AI Interface

Connect the headless game engine to external Large Language Models. Write serialization functions to format the game state into the strict JSON payload. Write parser functions to accept AI output, validate commands against the catalog, verify budget constraints, and execute grid placements. Integrate LiteLLM to pass the state to a test model and feed the response back to the parser to complete a continuous simulation loop.

### Phase Four Visualization

Develop visual debugging tools to verify engine physics and AI behavior. Map the NumPy temperature array to a visual color gradient where blue represents cold and red represents hot. Draw hardware sprites or rectangles indicating facing direction. Display real-time telemetry including current tick, bank balance, and PUE on the screen during the simulation.

### Phase Five Benchmark Suite

Finalize the project as an automated scientific evaluation tool. Introduce dynamic variables such as electricity price spikes or hardware failures to test agent adaptability. Write execution scripts to run identical simulation seeds simultaneously across different LLMs. Export final scores to CSV format and generate the comparative leaderboard.

