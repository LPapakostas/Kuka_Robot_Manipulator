# Kuka_Robot_Manipulator

NTUA Master Robotics Contol System Semester Exersice

## Installation

```bash
pip3 install virualenv
virtualenv robotics
source robotics/bin/activate
pip3 install -r requirements.txt
pip install .
```

## Usage

For kinematic simulation generation, run this command

```bash
python3 kuka_manipulator/simulation.py
```

For various plot generation, run this command  

```bash
python3 kuka_manipulator/plots.py
```

For trajectory animation, run this command

```bash
python3 kuka_manipulator/animation.py
```