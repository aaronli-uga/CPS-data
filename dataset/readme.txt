As described in the testbed architecture. Four Raspberry Pis (1-4) are able to wirelessly reach the router that connects the 4 controllers hooking up with the motors (1-4).

There are four sessions during the simulation as described in the figure. 3 Types of data are collected, namely, cyber data (network traffic in the network), physical data (the PMU data of the mini motors) and systematic data (systematic information of Raspberry Pis).

Two levels of labels are used to describe the data at each timestamp for the purpose of binary classification and multiclass classification. class_1 labels data as (1) normal and (2) attack. class_2 labels data as (1) normal, (2) x1_speedup (3) x2_speedup (4) motors_spike.

All attacks start with stepstone attack. It launchs from Raspberry Pi2, going through Pi3, Pi1 and Pi4. The actual attack controlling the motor speed is launched by Pi4 through MQTT command.
At each step, specifically, Raspberry Pi does the port scanning first to get the target Raspberry IP address and use brutal force attack to guess the ssh password of the target Raspberry Pi till getting the control of it. In this process, network traffic will surge due to port scanning and systematic data will (cpu, network, etc) due to the brute force attack.
