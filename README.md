# NeuGaze: Facial Expression & Gaze-Based Computer Control

**[中文文档](README-CN.md) | English**

[![arXiv](https://img.shields.io/badge/arXiv-2504.15101-b31b1b.svg)](https://arxiv.org/abs/2504.15101)
[![Demo Video](https://img.shields.io/badge/Demo-Bilibili-00A1D6)](https://www.bilibili.com/video/BV1kKdYYVEEM/#reply270100925344)

A non-invasive computer control system that combines facial expression recognition, head movement tracking, and gaze estimation, designed for hands-free human-computer interaction.

## Overview

Traditional assistive technologies face significant limitations: invasive brain-computer interfaces like Neuralink require surgical implantation, commercial eye trackers like Tobii lack precision for complex operations, and traditional assistive devices often involve cumbersome controls. NeuGaze addresses these challenges by integrating facial expressions, head movements, and gaze estimation to create an intuitive, hands-free control system.

**Requirements:**

- 📷 Standard webcam (no special hardware needed)
- 💻 CPU-only operation (no GPU required)
- 🪟 Currently supports Windows (other platforms may work but untested)

**Key Features:**

- 🎯 **Gaze-based mouse control** with real-time calibration
- 😊 **Facial expression mapping** to keyboard/mouse actions
- 🎮 **Three-modal control** combining gaze, expressions, and head movements
- ⚙️ **Customizable configurations** for different use cases
- 🚀 **Real-time performance** optimized for CPU inference

## Performance Evaluation

We conducted comprehensive testing using progressive training on multiple calibration datasets. The system achieves stable gaze tracking performance with mean errors of approximately **430 pixels** (about **48mm**) on a 3072×1920 display after training on multiple sessions.

![Progressive Training Analysis](results/progressive_training_analysis_english/comprehensive_comparison.png)

*Progressive training results showing error reduction and performance stability across multiple datasets. The analysis demonstrates consistent improvement in gaze accuracy as more training data is incorporated.*

## Installation

### Environment Setup

```bash
conda create -n neugaze python=3.11.11
conda activate neugaze
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install mediapipe==0.10.14 opencv-python==4.10.0.84
pip install jsonlines keyboard pywin32 filterpy==1.4.5 onnxruntime==1.21.1 tqdm==4.67.1
pip install PySide6 scikit-learn==1.6.1 ncnn pnnx pyyaml pyautogui
```

## Quick Start

### 1. Launch the GUI

```bash
python config_gui_cpu.py
```

### 2. Camera Setup

- Select your camera (default: camera 0)
- Position your face in the center of the preview window
- Click "Confirm Selection"

![Camera Selection](assets/camera%20selection.png)

### 3. Calibration

- Click "Start Calibration"
- Follow the on-screen dots with your gaze
- Keep your eyes open (photos are only taken when eyes are detected)
- Wait for calibration to complete

### 4. Start Control

- Click "Start Evaluation"
- Your mouse cursor will now follow your gaze
- Use facial expressions to trigger actions

## Expression & Control Configuration

NeuGaze uses a sophisticated expression recognition system defined in `configs/cpu.yaml`. The system supports multiple control modes and customizable mappings.

### Expression Detection

The system recognizes facial expressions through MediaPipe landmarks and maps them to specific actions:

#### Core Expressions

- **Open Mouth (`jawOpen`)**: *Drop your jaw naturally* → Mode selector - displays available control wheels
- **Pucker Lips (`mouthPucker`)**: *Make a kissing motion with pursed lips* → Left mouse click
- **Jaw Left (`jawLeft`)**: *Shift your jaw to the left side* → Right mouse click
- **Jaw Right (`jawRight`)**: *Shift your jaw to the right side* → Middle mouse click
- **Smile Left (`mouthSmileLeft`)**: *Smile with only the left side of your mouth* → Navigation/selection
- **Smile Right (`mouthSmileRight`)**: *Smile with only the right side of your mouth* → Navigation/selection
- **Both Sides Smile**: *Full natural smile with both sides* → Special commands
- **Head Movements**: *Tilt, turn, and nod your head* → WASD keys and scrolling

#### Expression Configuration Example

```yaml
left_click:
  conditions:
  - feature: mouthPucker
    operator: '>'
    threshold: 0.97
  - feature: mouthFunnel
    operator: <
    threshold: 0.2
  combine: AND
```

### Control Modes

The system supports multiple operation modes through the wheel interface:

#### 1. Game Mode (`game`)

Optimized for gaming with WASD movement and common game keys:

- **num1**: Z/X/C keys (common game actions)
- **num2**: Shift (sprint/crouch)
- **num4**: Number keys 1-4 (weapon selection)
- **num6**: Q/R/F/T keys (interaction keys)
- **num8**: Space (jump)

#### 2. CS:GO Mode (`game_cs`)

Specialized for Counter-Strike with tactical bindings:

- **num2**: Space (jump)
- **num8**: Shift (walk/precision)
- **Mouse lock**: Disabled for precise aiming

#### 3. Honor of Kings Mode (`game_wz`)

Optimized for MOBA gameplay (王者荣耀/Arena of Valor):

- **num1-3**: Skill activation (skills 1-3)
- **num4**: Map (M key)

#### 4. Typing Mode (`type`)

Full keyboard access for text input:

- **num4**: Complete alphabet in square layout
- **num6**: Numbers and symbols in square layout
- **num2**: Modifier keys (Shift, Ctrl, Alt, etc.)
- **num3**: Common shortcuts (Ctrl+C, Ctrl+V, etc.)

### Advanced Configuration

#### Expression Priorities

The system includes priority rules to prevent conflicting expressions:

```yaml
priority_rules:
- when: num7
  disable: [num2]
  except: []
```

#### Wheel Layouts

Different input modes support different wheel layouts:

- **Default**: Circular arrangement
- **Square**: Grid layout for alphabets and symbols (`layout_type: square`)

#### Head Movement Integration

Head orientation controls additional functions:

- **Pitch (up/down)**: W/S keys
- **Yaw (left/right)**: A/D keys
- **Roll (tilt)**: Scroll wheel

## Use Cases

### Accessibility

- **Mobility Assistance**: Hands-free computer operation for users with limited mobility
- **Rehabilitation**: Motor skill training through controlled head and facial movements

### Gaming & Entertainment

- **Immersive Gaming**: Novel input method for enhanced gaming experiences
- **Muscle Training**: Facial and neck muscle exercise through interactive control

### Smart Device Integration

- **AR/VR Interfaces**: Natural control for head-mounted displays
- **Smart Glasses**: Expression-based navigation without hand gestures

## Technical Details

### Architecture

- **Intent Recognition**: Comprehensive analysis of facial expressions, head movements, and gaze patterns
- **Intent Mapping**: Translation of recognized intents into specific keyboard/mouse actions
- **Multi-Modal Fusion**: Integration and prioritization of multiple simultaneous intents
- **Action Execution**: Coordinated control system enabling complex gaming operations
- **Optimization**: CPU-optimized inference pipeline for real-time performance

### Limitations

- **Lighting Sensitivity**: Performance degrades in poor or uneven lighting
- **Calibration Required**: Individual calibration needed for optimal accuracy
- **Expression Training**: Learning curve for natural expression control

## Contributing

We welcome contributions! Please feel free to submit issues, feature requests, or pull requests.

## Citation

```bibtex
@article{yang2024neugaze,
  title={NeuGaze: Facial Expression and Gaze-Based Computer Control},
  author={Yang, Yiqian},
  journal={arXiv preprint arXiv:2504.15101},
  year={2024}
}
```

## License

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

This project is licensed under Creative Commons Attribution-NonCommercial 4.0 International License.

**You are free to:**

- ✅ Share and adapt the code for personal use
- ✅ Use for research and educational purposes
- ✅ Create derivative works for non-commercial purposes (e.g., streaming, content creation)

**You may NOT:**

- ❌ Sell the source code or derivatives
- ❌ Deploy as commercial hardware/software products
- ❌ Package as paid executable applications
- ❌ Use for commercial web services

We provide this software freely to benefit the community while preventing exploitation by commercial entities. License terms may be updated to Apache-2.0 or MIT based on community feedback.

---

**Note**: This system is designed for research and accessibility purposes. While functional, it may require individual tuning for optimal performance. We encourage experimentation and welcome feedback to improve the system's robustness and usability.
