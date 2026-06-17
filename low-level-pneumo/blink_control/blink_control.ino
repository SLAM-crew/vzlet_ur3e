const int relayPin = 2;

void setup() {
  // Configure the relay pin and the built-in LED as outputs
  pinMode(relayPin, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);

  // Ensure the relay and LED are off when starting up
  digitalWrite(relayPin, LOW);
  digitalWrite(LED_BUILTIN, LOW);
}

void loop() {
  // Turn the relay (and valve) ON, and turn on the built-in LED
  digitalWrite(relayPin, HIGH);
  digitalWrite(LED_BUILTIN, HIGH);
  delay(3000); // Wait for 3 seconds

  // Turn the relay (and valve) OFF, and turn off the built-in LED
  digitalWrite(relayPin, LOW);
  digitalWrite(LED_BUILTIN, LOW);
  delay(3000); // Wait for 3 seconds
}
