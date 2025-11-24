# IBX Land Value Project

Code + notebooks for:
- transit walkshed tessellation
- served census / DOTS
- land value estimation and uplift


Pipeline outline:
- Download the property value datasets from OpenNY (do this manually)
- Draw a line that tracks proposed paths for the IBX (make this modular so that it can also track other lines) as some sort of interpretable geometry over a map of NYC
- Expand that line to cover .5 miles on both sides (CHECK THIS AGAINST GUPTA PAPER)
- Turn the csv rows for relevant buildings (buildings roughly within the region) into geojsons 
- Pull every building that is within the rough .5m walkshed and their values
- Exclude every building in that sample that is not eligible for the law's value capture mechanism (public land etc.)
- Sum the value and print it, along with a map of the proposed walkshed. Multiply the number by 1.04, 1.06, 1.08, 1.1 to estimate different value uplift scenarios.
- Using the Gupta paper as a model, estimate what tax rates would lead to appropriate value capture over what period of time.

