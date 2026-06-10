#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${SCRIPT_DIR}/ws"
COMPOSE_FILE="${SCRIPT_DIR}/docker/ur3e.compose.yaml"
IMAGE_NAME="crew_docker-ur3e:latest"

usage() {
  cat <<'EOF'
Usage:
  ./ctl.sh image-build
  ./ctl.sh workspace-init
  ./ctl.sh workspace-build
  ./ctl.sh workspace-clean
  ./ctl.sh build
  ./ctl.sh up
  ./ctl.sh down
  ./ctl.sh exec
EOF
}

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

build_image() {
  compose build ur3e
}

workspace_init() {
  build_image

  if [ -d "${WORKSPACE_DIR}/src" ] && find "${WORKSPACE_DIR}/src" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
    echo "workspace-init skipped: ws/src is not empty" >&2
    return 0
  fi

  sudo docker run --rm -it \
    -v "${WORKSPACE_DIR}:/root/workspace" \
    -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
    "${IMAGE_NAME}" \
    bash -lc 'set -euo pipefail; cd /root/workspace; mkdir -p src; git clone -b kilted https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git src/Universal_Robots_ROS2_Driver; vcs import src --skip-existing --input src/Universal_Robots_ROS2_Driver/Universal_Robots_ROS2_Driver.kilted.repos; rosdep init 2>/dev/null || true; rosdep update; rosdep install --ignore-src --from-paths src -y --rosdistro kilted'
}

workspace_build() {
  build_image
  sudo docker run --rm -it \
    -v "${WORKSPACE_DIR}:/root/workspace" \
    -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
    "${IMAGE_NAME}" \
    bash -lc 'source /opt/ros/kilted/setup.bash && cd /root/workspace && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release'
}

workspace_clean() {
  build_image
  sudo docker run --rm -it \
    -v "${WORKSPACE_DIR}:/root/workspace" \
    -e RMW_IMPLEMENTATION=rmw_zenoh_cpp \
    "${IMAGE_NAME}" \
    bash -lc 'cd /root/workspace && rm -rf build install log'
}

cmd="${1:-}"
case "${cmd}" in
  image-build)
    build_image
    ;;
  workspace-init)
    workspace_init
    ;;
  workspace-build)
    workspace_build
    ;;
  workspace-clean)
    workspace_clean
    ;;
  build)
    workspace_build
    ;;
  up)
    compose up -d ur3e
    ;;
  down)
    compose down
    ;;
  exec)
    compose exec ur3e bash
    ;;
  *)
    usage
    exit 1
    ;;
esac
