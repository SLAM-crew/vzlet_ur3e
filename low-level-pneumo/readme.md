
Install arduino-cli && this extension
```
arduino-cli core install arduino:avr
```
Flash:
```
arduino-cli compile --upload -p /dev/ttyUSB0 --fqbn arduino:avr:nano blink
```
