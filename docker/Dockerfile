# syntax=docker/dockerfile:1

# ROS 2 Jazzy + PX4 + Micro XRCE-DDS Agent development image.
#
# Scope of this draft:
#   - ROS 2 Jazzy / Ubuntu 24.04
#   - Python 3.12
#   - PX4-Autopilot v1.17.0 SITL / Gazebo Harmonic
#   - Micro XRCE-DDS Agent v2.4.3
#
# Deliberately excluded for now:
#   - MAPPO runtime dependencies, including PyTorch
#   - MAVROS
#   - YOLO, Ultralytics, OpenCV and the yolo_infer Conda environment

FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG USERNAME=ncrl
ARG USER_UID=1000
ARG USER_GID=${USER_UID}

ARG PX4_VERSION=v1.17.0
ARG MICRO_XRCE_DDS_AGENT_VERSION=v2.4.3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    ROS2_WS=/home/${USERNAME}/mappo_ros2 \
    PX4_DIR=/home/${USERNAME}/PX4-Autopilot \
    MICRO_XRCE_DDS_AGENT_DIR=/home/${USERNAME}/Micro-XRCE-DDS-Agent

# ROS 2 development and workspace tools. PX4's own ubuntu.sh installs the
# additional compiler, Python, GStreamer and Gazebo dependencies it requires.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        sudo \
        tmux \
        vim \
        wget \
        python3-colcon-common-extensions \
        python3-dev \
        python3-pip \
        python3-rosdep \
        python3-vcstool \
        ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Non-root development user. Ubuntu 24.04 images may already reserve UID/GID
# 1000 for an "ubuntu" account. Reuse and rename those numeric IDs when they
# exist; otherwise create a new account. This keeps bind-mount ownership aligned
# with the host user without failing on an occupied UID or GID.
RUN set -eux; \
    existing_group="$(getent group "${USER_GID}" | cut -d: -f1 || true)"; \
    if getent group "${USERNAME}" >/dev/null; then \
        true; \
    elif [ -n "${existing_group}" ]; then \
        groupmod --new-name "${USERNAME}" "${existing_group}"; \
    else \
        groupadd --gid "${USER_GID}" "${USERNAME}"; \
    fi; \
    existing_user="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
        true; \
    elif [ -n "${existing_user}" ]; then \
        usermod --login "${USERNAME}" \
                --home "/home/${USERNAME}" \
                --move-home \
                "${existing_user}"; \
    else \
        useradd --uid "${USER_UID}" \
                --gid "${USERNAME}" \
                --create-home \
                --shell /bin/bash \
                "${USERNAME}"; \
    fi; \
    usermod --gid "${USERNAME}" --shell /bin/bash "${USERNAME}"; \
    echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" \
        > "/etc/sudoers.d/${USERNAME}"; \
    chmod 0440 "/etc/sudoers.d/${USERNAME}"

RUN mkdir -p "/home/${USERNAME}/mappo_ros2" \
    && chown -R "${USERNAME}:${USERNAME}" "/home/${USERNAME}/mappo_ros2"

# This Dockerfile expects the repository root as the Docker build context.
COPY --chown=${USERNAME}:${USERNAME} \
    docker/tmux.conf \
    /home/${USERNAME}/.tmux.conf

# Initialize ROS 2 for both interactive and non-interactive container commands.
COPY --chown=root:root --chmod=0755 \
    docker/entrypoint.sh \
    /usr/local/bin/ros_entrypoint.sh

# Interactive terminals automatically receive ROS 2 and the overlay workspace
# when it has been built.
RUN echo 'source /opt/ros/jazzy/setup.bash' \
        >> "/home/${USERNAME}/.bashrc" \
    && echo '[ -f "$HOME/mappo_ros2/install/setup.bash" ] && source "$HOME/mappo_ros2/install/setup.bash"' \
        >> "/home/${USERNAME}/.bashrc"

# PX4-Autopilot is kept beside mappo_ros2 under /home/ncrl. The setup script
# installs SITL and Gazebo Harmonic dependencies; --no-nuttx omits the Pixhawk
# firmware cross-toolchain for now.
WORKDIR /home/${USERNAME}

RUN git clone \
        --branch "${PX4_VERSION}" \
        --depth 1 \
        --recurse-submodules \
        --shallow-submodules \
        https://github.com/PX4/PX4-Autopilot.git \
        "${PX4_DIR}"

# Apply the one-line encoder selection change from PX4 PR #27944 when the
# pinned release still uses nvh264enc. Newer sources that already contain the
# change pass through unchanged.
RUN set -eux; \
    gst_camera_file="${PX4_DIR}/src/modules/simulation/gz_plugins/gstreamer/GstCameraSystem.cpp"; \
    if grep -q 'gst_element_factory_make("nvh264enc"' "${gst_camera_file}"; then \
        sed -i \
            's/gst_element_factory_make("nvh264enc"/gst_element_factory_make("nvautogpuh264enc"/' \
            "${gst_camera_file}"; \
    fi; \
    grep -q 'gst_element_factory_make("nvautogpuh264enc"' "${gst_camera_file}"

RUN RUNS_IN_DOCKER=true \
        bash "${PX4_DIR}/Tools/setup/ubuntu.sh" --no-nuttx \
    && rm -rf /var/lib/apt/lists/* \
    && chown -R "${USERNAME}:${USERNAME}" "${PX4_DIR}"

# PX4 v1.17 recommends Micro XRCE-DDS Agent v2.4.3 for ROS 2
# communication. Install the executable and shared libraries under /usr/local;
# retain the pinned source beside PX4 for inspection or later rebuilding.
RUN git clone \
        --branch "${MICRO_XRCE_DDS_AGENT_VERSION}" \
        --depth 1 \
        https://github.com/eProsima/Micro-XRCE-DDS-Agent.git \
        "${MICRO_XRCE_DDS_AGENT_DIR}" \
    && cmake \
        -S "${MICRO_XRCE_DDS_AGENT_DIR}" \
        -B "${MICRO_XRCE_DDS_AGENT_DIR}/build" \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build "${MICRO_XRCE_DDS_AGENT_DIR}/build" \
        --parallel "$(nproc)" \
    && cmake --install "${MICRO_XRCE_DDS_AGENT_DIR}/build" \
    && ldconfig \
    && rm -rf "${MICRO_XRCE_DDS_AGENT_DIR}/build" \
    && chown -R "${USERNAME}:${USERNAME}" \
        "${MICRO_XRCE_DDS_AGENT_DIR}"

USER ${USERNAME}
WORKDIR /home/${USERNAME}/mappo_ros2

# Verify ROS 2, PX4's Python requirements, Gazebo, Micro XRCE-DDS Agent and the
# PR #27944 source change. GPU/NVENC availability itself can only be tested at
# container runtime.
RUN source /opt/ros/jazzy/setup.bash \
    && python3 -m pip check \
    && command -v gz >/dev/null \
    && command -v MicroXRCEAgent >/dev/null \
    && test -d "${PX4_DIR}/src" \
    && grep -q 'gst_element_factory_make("nvautogpuh264enc"' \
        "${PX4_DIR}/src/modules/simulation/gz_plugins/gstreamer/GstCameraSystem.cpp" \
    && python3 - <<'PY'
import rclpy

print("rclpy import: OK")
PY

ENTRYPOINT ["/usr/local/bin/ros_entrypoint.sh"]
CMD ["bash"]
