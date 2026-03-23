# Pipeline Diagrams

## Training Pipeline

```mermaid
flowchart TD
    A[Player Stats + Game Logs] --> B[get_training_matrix]
    B --> C[Feature Matrix X, Targets y]
    C --> D[Train / Validation / Test Split]

    D --> E[LightGBMLSS Training]
    E --> F["Distribution Params<br/>(r, p, alpha, beta, gate)"]

    D --> G[Book EVs from Archive]

    F --> H[fit_model_weight<br/>on Validation Set]
    G --> H
    H --> I["Optimal Weight w"]

    F --> J["fused_loc<br/>(blend model + book params)"]
    G --> J
    I --> J
    J --> K[Blended Distribution]

    K --> L["get_odds<br/>(CDF at line)"]
    L --> M["Raw P(over), P(under)"]

    M --> N["Temperature Scaling<br/>Fit T >= 1 minimizing Brier score"]
    N --> O["T_opt"]

    O --> P["Apply to Test Set<br/>p_cal = sigmoid(logit(p) / T)"]
    M --> P
    P --> Q["Calibrated Probabilities"]

    Q --> R["Compute Metrics<br/>(Accuracy, Precision, Sharpness, NLL)"]
    R --> S["Save Model Pickle<br/>(model, temperature, weight, cv, ...)"]

    style N fill:#f9d77e,stroke:#d4a017
    style O fill:#f9d77e,stroke:#d4a017
    style P fill:#f9d77e,stroke:#d4a017
```

## Inference Pipeline

```mermaid
flowchart TD
    A[Sportsbook Offers<br/>DraftKings, FanDuel, PrizePicks, ...] --> B[Load Model Pickle]
    B --> C["Model, Temperature T, Weight w, CV"]

    A --> D["get_stats per player<br/>(feature vector)"]
    D --> E[LightGBMLSS Predict]
    E --> F["Distribution Params<br/>(r, p, alpha, beta, gate)"]

    A --> G[Book EVs from Archive]

    F --> H["fused_loc<br/>(blend model + book params)"]
    G --> H
    C --> H
    H --> I[Blended Distribution]

    I --> J["get_odds<br/>(CDF at line)"]
    J --> K["Raw P(over), P(under)"]

    K --> L["Temperature Scaling<br/>p_cal = sigmoid(logit(p) / T)"]
    C --> L
    L --> M["Calibrated P(over), P(under)"]

    M --> N["Hard Cap at 90%"]
    N --> O["Model P = max(P_over, P_under)<br/>Bet = argmax direction"]

    O --> P["Apply Boost for Kelly<br/>Model = Model_P * Boost<br/>K = (Model - 1) / (Boost - 1)"]
    P --> Q[Output to Google Sheets]

    style L fill:#f9d77e,stroke:#d4a017
    style N fill:#f9d77e,stroke:#d4a017
    style O fill:#a8d5a2,stroke:#4a9e3f
    style P fill:#a8d5a2,stroke:#4a9e3f
```

### Legend

- Yellow: Calibration steps (temperature scaling, confidence cap)
- Green: Confidence and bet sizing (pure probability kept separate from boost)
