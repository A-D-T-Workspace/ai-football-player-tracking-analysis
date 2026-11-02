# Engineering Case Study: AI Football Player Tracking & Performance Analysis

**Project Category:** Computer Vision / Machine Learning Engineering  
**Project Type:** Applied AI Research and Development  
**Status:** Functional MVP  
**Primary Technologies:** Python, YOLO11n, ByteTrack, OpenCV, NumPy, Pandas, Matplotlib, Pytest

**Repository:** https://github.com/A-D-T-Workspace/ai-football-player-tracking-analysis

---

# 1. Executive Summary

This project explores the development of a computer vision system capable of detecting, tracking, and analysing football players from broadcast match footage.

The objective was to investigate how modern object detection and multi-object tracking techniques could be combined with movement analysis to extract structured information about individual players.

The system uses YOLO11n for player detection and ByteTrack for multi-object tracking. Tracking observations are subsequently processed to estimate player movement, evaluate track quality, identify fragmented trajectories, and generate visual analytics.

A series of controlled experiments was conducted to investigate the effect of different tracking configurations on tracking continuity.

The project produced a functional MVP with:

- Automated player detection and tracking.
- Configurable multi-object tracking experiments.
- Tracking continuity diagnostics.
- Player trajectory extraction.
- Image-plane movement and speed estimation.
- Track quality classification.
- Structured CSV outputs.
- Automated tests for core movement-analysis functionality.

An important engineering finding was that obtaining detections is not equivalent to maintaining reliable player identities over time. Occlusion, camera movement, and missed detections can fragment trajectories and undermine downstream performance analysis.

The current implementation establishes a foundation for more advanced football analytics, while identifying pitch calibration, identity persistence, and ground-truth evaluation as major areas for further development.

# 2. The Problem

Football clubs and performance analysts increasingly rely on player tracking data to understand individual and team performance.

Examples of valuable information include:

- Distance covered.
- Average and maximum speed.
- Sprint frequency.
- Acceleration and deceleration.
- Positional distribution.
- Time spent in different pitch zones.
- Movement patterns.
- Tactical positioning.

However, obtaining this information can require specialised tracking cameras, expensive infrastructure, or commercial analytics platforms.

This creates an accessibility challenge for smaller clubs, academies, and football organisations, particularly in developing football markets.

The engineering question behind this project was:

**Can a computer vision pipeline extract useful player movement information from ordinary football broadcast footage using accessible machine learning technologies?**

The project investigated this question by building a modular detection, tracking, and movement-analysis pipeline.

# 3. Engineering Objectives

The project was developed around eight objectives.

1. Detect football players in broadcast footage.
2. Associate detections across video frames using multi-object tracking.
3. Investigate player identity continuity and track fragmentation.
4. Extract individual player trajectories.
5. Estimate movement displacement and speed.
6. Evaluate track quality before downstream analysis.
7. Produce structured datasets and visualisations.
8. Establish a foundation for real-world pitch calibration and advanced performance analytics.

The emphasis was not simply on running a pretrained detection model. It was on understanding how errors in one stage of a computer vision pipeline affect subsequent analytical stages.

# 4. System Architecture

The system follows a modular processing architecture.

```text
             Football Broadcast Video
                       |
                       v
             Video Inspection
                       |
                       v
                Frame Sampling
                       |
                       v
                YOLO11n Detection
                       |
                       v
              ByteTrack Association
                       |
                       v
              Tracking Observations
                       |
                       v
             Track Continuity Analysis
                       |
                       v
              Movement Computation
                       |
                       v
              Track Quality Assessment
                       |
                       v
              Trajectory Visualisation
                       |
                       v
             Structured Analytics Output
```

## 4.1 Video Processing

The input footage is inspected to determine its resolution, frame rate, duration, and frame count.

The experimental video had the following properties:

| Property | Value |
|---|---:|
| Resolution | 640 × 360 |
| Original frame rate | 30 FPS |
| Total frames | 162,565 |
| Duration | Approximately 90.3 minutes |

For the initial tracking experiments, a 10-second segment was sampled at 15 FPS.

This produced 150 sampled frames.

Sampling reduced the computational workload while providing a controlled environment for comparing tracker configurations.

## 4.2 Player Detection

YOLO11n was used to detect people in football footage.

The detection stage produces bounding boxes and confidence scores that are passed to the tracking stage.

The decision to begin with a pretrained lightweight model made it possible to develop and test the pipeline without requiring a large custom training dataset.

## 4.3 Multi-Object Tracking

ByteTrack was used to associate detections across consecutive sampled frames.

Each tracked object receives an identifier, allowing its observed positions to be grouped into trajectories.

The tracker configuration was modified during experimentation to investigate whether a longer track buffer could improve continuity when detections temporarily disappeared.

## 4.4 Tracking Data Representation

Tracking observations were stored in CSV format.

The observation dataset contains:

| Field | Description |
|---|---|
| frame | Sampled video frame index |
| track_id | Assigned tracking identifier |
| x1 | Bounding-box left coordinate |
| y1 | Bounding-box top coordinate |
| x2 | Bounding-box right coordinate |
| y2 | Bounding-box bottom coordinate |
| confidence | Detection confidence |

This representation separates detection and tracking from downstream analytics.

It also makes it possible to reproduce movement analysis without rerunning the entire detection pipeline.

# 5. Major Engineering Challenges

## 5.1 Player Identity Fragmentation

One of the most important challenges was maintaining consistent player identities.

A player can remain visible in the footage while the tracking algorithm assigns a new identifier.

This can occur because of:

- Player-to-player occlusion.
- Missed detections.
- Similar player appearances.
- Fast movement.
- Camera movement.
- Changes in bounding-box position or appearance.

The result is trajectory fragmentation.

A player who should have one continuous trajectory may instead be represented by several shorter tracking IDs.

This matters because movement calculations depend on the quality of the underlying trajectory.

## 5.2 Tracking Configuration Optimisation

Rather than relying on one tracker configuration, multiple experiments were conducted.

The objective was to compare tracking behaviour under different confidence thresholds and track-buffer settings.

This introduced a more systematic experimental approach to the tracking problem.

## 5.3 Camera Movement and Pitch Geometry

Broadcast football footage frequently contains camera panning, zooming, and changes in perspective.

These movements complicate the interpretation of player positions.

A player can appear to move across the image even when their actual movement relative to the pitch is smaller.

To investigate this challenge, additional computer vision experiments explored:

- Pitch edge detection.
- Line segment extraction.
- Candidate line intersections.
- Feature matching.
- RANSAC-based camera diagnostics.

These experiments helped investigate the geometry of the footage, but did not produce a validated pitch calibration system.

## 5.4 Movement Measurement Reliability

Another important challenge was distinguishing image-plane movement from real-world movement.

Pixel displacement alone cannot establish how many metres a player has covered.

The apparent size and movement of a player depend on perspective and camera geometry.

The MVP therefore reports movement in pixels and speed in pixels per second rather than presenting unvalidated real-world measurements.

# 6. Experimental Methodology

Three tracking configurations were evaluated using the same 10-second sampled video segment.

The experiment used:

- 150 sampled frames.
- 15 FPS sampling.
- The same detection model.
- Different confidence and tracking configurations.

## 6.1 Experimental Comparison

| Experiment | Confidence | Tracker | Observations | Unique IDs | Visible ID-set Changes |
|---|---:|---|---:|---:|---:|
| Baseline | 0.10 | Default ByteTrack | 1,057 | 54 | 101 |
| Confidence 0.25 | 0.25 | Default ByteTrack | 823 | 62 | 127 |
| Buffer 60 | 0.10 | Custom ByteTrack | 1,170 | 36 | 89 |

## 6.2 Selected Configuration

The custom configuration used:

```yaml
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: 60
match_thresh: 0.8
fuse_score: True
```

The Buffer 60 experiment produced:

- 1,170 tracking observations.
- 36 unique tracking IDs.
- 89 visible ID-set changes.
- 75 new-ID appearances.
- 4 short tracks containing 1–3 observations.
- 22 long tracks containing at least 10 observations.
- Longest track: 106 observations.

## 6.3 Interpretation

The custom Buffer 60 configuration produced fewer unique IDs and fewer visible ID-set changes than the baseline.

It also produced more tracking observations.

These results suggest improved continuity-related behaviour in this particular experiment.

However, they do not prove that the tracker achieved a higher identity accuracy.

The evaluation did not use manually annotated ground-truth player identities. Therefore, visible ID-set changes and new-ID appearances are treated as diagnostic proxies rather than verified ID-switch counts.

This distinction is important when interpreting multi-object tracking experiments.

# 7. Player Movement Analysis

After tracking, the system processes player observations to estimate movement.

The movement-analysis module uses bounding-box foot positions as the representative player location.

This provides a more useful approximation of ground contact than the bounding-box centre for analysing player movement across the image.

## 7.1 Extracted Metrics

The system calculates:

| Metric | Description |
|---|---|
| Track duration | Time between the first and last observations |
| Total movement | Accumulated displacement |
| Average speed | Average image-plane movement speed |
| Maximum speed | Maximum estimated image-plane movement speed |
| Observation count | Number of observations associated with a track |
| Track quality | Classification based on observation continuity and duration |

## 7.2 Track Quality Assessment

The selected tracking experiment produced 36 unique tracks.

The movement-analysis stage classified them as:

| Classification | Tracks |
|---|---:|
| Usable | 16 |
| Short or fragmented | 20 |
| Total | 36 |

This demonstrated why tracking quality assessment is necessary before extracting meaningful performance metrics.

Not every detected or tracked object provides enough reliable observations for useful movement analysis.

## 7.3 Measurement Units

The current implementation calculates:

- Displacement in pixels.
- Average speed in pixels per second.
- Maximum speed in pixels per second.

These measurements are useful for investigating movement patterns within the image.

They are not equivalent to physical distance or speed on the football pitch.

# 8. Pitch and Camera Diagnostics

A separate investigation examined whether football pitch markings could support geometric calibration.

The pitch-analysis work included edge detection, line extraction, and intersection candidate generation.

One experiment produced:

| Diagnostic | Result |
|---|---:|
| Long line segments | 77 |
| Raw intersection candidates | 167 |
| Unique candidate intersections | 49 |
| Candidates inside central image region | 15 |

These intersections were candidates only. They were not confirmed pitch landmarks.

Camera diagnostics also explored ORB feature matching and RANSAC-based estimation.

The investigation demonstrated that camera movement and changing visual conditions complicate the use of a single fixed image-to-pitch transformation.

A validated homography and real-world coordinate system remain future work.

# 9. Testing and Software Reliability

The project includes automated tests for core movement-analysis functionality.

The test suite validates:

1. Foot-position calculation.
2. Movement displacement calculation.
3. Exclusion of large observation gaps.
4. Elapsed-time calculations using original video FPS.
5. Track quality thresholds.

The latest recorded test result was:

```text
5 passed
```

The tests help protect important movement-analysis logic against regressions.

However, these tests do not establish the accuracy of the YOLO detector or ByteTrack identity assignments.

## Reproducibility

The project separates tracking observations from downstream analysis.

This allows analysis scripts to reuse previously generated tracking data without repeating expensive video inference.

The experiment log also records tracker configurations and selected tracking statistics to support comparisons.

# 10. Key Engineering Findings

The project produced several useful engineering findings.

### Finding 1 — Detection and tracking are different problems

A detector can identify players in individual frames without guaranteeing consistent identities across time.

Tracking requires reliable association between observations.

### Finding 2 — Tracker configuration influences continuity

Changes to confidence thresholds and track-buffer settings can substantially affect the number of observations and track identifiers produced.

The effect must be measured rather than assumed.

### Finding 3 — Tracking quality affects movement analytics

Fragmented trajectories can produce incomplete or misleading movement estimates.

Track quality assessment should therefore be part of the analytics pipeline.

### Finding 4 — Camera geometry matters

Image-plane movement is not equivalent to movement across the real football pitch.

Camera calibration is necessary for defensible real-world distance and speed estimates.

### Finding 5 — Evaluation methodology matters

A reduction in tracking IDs or visible ID-set changes does not automatically establish improved identity accuracy.

Reliable evaluation requires ground-truth annotations and appropriate tracking metrics.

# 11. Potential Applications in Football

A more mature version of this system could support several areas of football operations.

## 11.1 Nigerian Football

Potential applications include:

- Player performance analysis for Nigerian clubs.
- Academy player development.
- Scouting and talent identification.
- Video-based tactical analysis.
- Player workload monitoring.
- Development of accessible football analytics infrastructure.

Lower-cost video analytics could help make performance analysis more accessible to clubs that cannot afford specialised tracking infrastructure.

## 11.2 African Football

Across African football leagues and academies, the approach could support:

- Standardised player performance data.
- Youth development programmes.
- Cross-club scouting.
- Player development comparisons.
- Research into football performance and tactics.

## 11.3 International Football

With improved tracking accuracy, calibration, and validation, the system could be developed further for:

- Professional club analytics.
- Player movement analysis.
- Tactical research.
- Performance benchmarking.
- Sports science research.

These are potential applications rather than outcomes already validated by the current MVP.

# 12. Limitations and Future Work

The project is an experimental MVP and has several limitations.

| Area | Current limitation | Future direction |
|---|---|---|
| Identity tracking | Player IDs can fragment | Re-identification and improved association |
| Pitch calibration | Not validated | Landmark-based homography |
| Movement metrics | Image-plane units | Real-world coordinate conversion |
| Sprint analysis | Not yet validated | Calibrated speed and acceleration thresholds |
| Heatmaps | Not yet implemented as pitch-space analytics | Pitch-coordinate occupancy |
| Evaluation | No ground-truth identity annotations | Manually annotated benchmark clips |
| Video processing | Initial experiments use short segments | Full-match inference and validation |
| Player selection | No complete persistent selection workflow | Player lock-on and identity persistence |

## Proposed Development Roadmap

**Phase 1 — Tracking reliability**

- Investigate player re-identification.
- Improve handling of occlusion.
- Develop ground-truth evaluation clips.
- Measure identity-switch and tracking metrics.

**Phase 2 — Pitch calibration**

- Detect reliable pitch landmarks.
- Estimate homography transformations.
- Validate calibration across camera movement.
- Convert image coordinates into pitch coordinates.

**Phase 3 — Advanced movement analytics**

- Distance covered in metres.
- Speed in metres per second.
- Sprint detection.
- Acceleration and deceleration.
- Positional heatmaps.
- Pitch-zone occupancy.

**Phase 4 — Full-match analysis**

- Extend processing to complete matches.
- Improve computational efficiency.
- Evaluate tracking quality across different camera conditions.
- Produce more comprehensive player performance datasets.

**Phase 5 — Scalable deployment**

- GPU-accelerated inference.
- Cloud-based processing.
- Reproducible model evaluation.
- Structured analytics outputs for downstream applications.

# 13. Conclusion

This project demonstrates the development of a modular computer vision pipeline for football player detection, multi-object tracking, and movement analysis.

The most important outcome was not simply producing tracking IDs or movement charts. It was investigating how detection, tracking continuity, camera geometry, and measurement reliability interact within a complete analytical system.

The experiments exposed practical limitations of multi-object tracking in broadcast footage and established a foundation for more rigorous evaluation and real-world performance measurement.

The current MVP provides a starting point for further work in football computer vision, sports analytics, and applied machine learning engineering.

---

**Project Repository:**  
https://github.com/A-D-T-Workspace/ai-football-player-tracking-analysis

**Project:** AI Football Player Tracking & Performance Analysis