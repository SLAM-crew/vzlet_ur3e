#!/bin/bash

# Define the rule file path
RULE_FILE="/etc/udev/rules.d/99-feetech.rules"

echo "Creating udev rule for Feetech Driver (CH340)..."

# The udev rule: matches the CH340 Vendor and Product ID, creates the symlink, and grants read/write permissions
RULE='SUBSYSTEM=="tty", ATTRS{idVendor}=="1a86", ATTRS{idProduct}=="7523", SYMLINK+="feetech_driver", MODE="0666"'

# Write the rule to the system directory (requires sudo)
echo "$RULE" | sudo tee $RULE_FILE > /dev/null

echo "Rule created at $RULE_FILE"

# Reload the udev rules and trigger them so you don't have to unplug the USB
echo "Reloading udev rules..."
sudo udevadm control --reload-rules
sudo udevadm trigger

echo "Setup complete! Verifying the symlink..."
sleep 1 # Wait a moment for udev to populate the symlink
ls -l /dev/feetech_driver