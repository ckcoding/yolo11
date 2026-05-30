# YOLO11 Visual Platform

This repository contains one deployable project with two main services:

- `stream_service/`: realtime detection API and stream rendering service.
- `mmyolo_small_object/`: MMYOLO small-object training platform with a web console.

Large model weights, datasets, runtime logs, generated training outputs, and virtual environments are intentionally excluded from Git. Put required weights under the documented local paths before deployment.

## Realtime Detection Service

Start the stream service from the repository root:

```bash
cd stream_service
docker compose up -d --build
```

The default API endpoint is:

```text
http://127.0.0.1:18008/models
```

Configure label names and box colors in:

```text
stream_service/config.yml
```

Example:

```yaml
labels:
  display_names:
    drone_people: "人员"
    drone_car: "车"
  colors:
    drone_people: "#00D1FF"
    drone_car: "#00FF66"
```

## Training Platform

The training platform is part of the same project under:

```text
mmyolo_small_object/
```

Start it from the repository root:

```bash
docker compose -f mmyolo_small_object/docker-compose.console.yml up -d --build
```

Then open:

```text
http://127.0.0.1:18080
```

The console writes runtime data to `mmyolo_small_object/runtime/`, including jobs, logs, prepared datasets, generated configs, and `work_dirs`. This directory is ignored by Git.

If you want RTMDet to use the default local pretrained checkpoint, place it here:

```text
mmyolo_small_object/cspnext-tiny_imagenet_600e.pth
```

The checkpoint is excluded from Git because it is a large binary artifact.
