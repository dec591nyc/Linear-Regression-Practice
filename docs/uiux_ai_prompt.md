# UI/UX Prompt for AI App Development

Use this prompt when asking an AI assistant to improve a Streamlit, dashboard, or data-analysis app UI.

```text
You are a senior product designer and frontend engineer. Improve this app as a real user-facing analytical tool, not as a decorative demo.

Context:
- App type: Streamlit data-analysis dashboard.
- Audience: non-technical users who need to understand a model result without knowing math.
- Domain: air quality, AQI, regression, residual/outlier explanation.
- Current problem: the UI is hard to read in light mode, controls feel like raw Streamlit widgets, and model metrics need plain-language explanations.

Design goals:
1. Prioritize readability and trust over decoration.
2. Use a clean light mode first, then adapt dark mode.
3. Keep typography restrained:
   - Main title: 28-34px equivalent.
   - Section headings: 18-22px.
   - Body text: 14-16px.
   - Metric values: visible but not oversized.
4. Make every control readable in both themes:
   - Sidebar text
   - Tabs
   - Selectbox
   - Number input
   - Slider labels
   - File/download buttons
   - Dataframes
5. Avoid using white text on light backgrounds or dark widgets embedded in light mode unless intentionally styled.
6. Replace purely textual toggles with recognizable icon + short label controls, such as language and theme buttons.
7. Do not add upload features unless the accepted schema is documented and validated.
8. Add a plain-language interpretation block under every chart:
   - What each dot/line/table row means.
   - What the metric means.
   - What action a user should take.
   - What not to overclaim.
9. For model metrics, explain:
   - R-squared: how much pattern the model captured.
   - RMSE/MAE: average error; lower is better.
   - Residual/outlier: where prediction and reality differ most.
10. Do not use landing-page hero design. This is a working analytical interface.

Deliverables:
- Revised UI structure.
- CSS/theme strategy.
- Component-level changes.
- Plain-language copy for metrics and charts.
- Any removed features with justification.
- Verification checklist for desktop light mode, desktop dark mode, and narrow viewport.

Before coding, state:
- The primary user workflow.
- The information hierarchy.
- Which controls are essential and which should be removed.

After coding, verify:
- No unreadable text in light mode.
- No duplicate Streamlit widget keys.
- App runs without StreamlitDuplicateElementId.
- The first screen explains what the app does without requiring math knowledge.
```
