const int relayPin = 2;
bool valveState = false;

void setup() {
  Serial.begin(115200);

  pinMode(LED_BUILTIN, OUTPUT);
  pinMode(relayPin, OUTPUT);

  // Start with valve and LED OFF
  digitalWrite(relayPin, LOW);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.println("System Ready.");
  Serial.println("Commands: 'pneumo1' (Open), 'pneumo0' (Close), 'status' (Check state)");
}

void loop() {
  // Check if data is available to read from the PC
  if (Serial.available() > 0) {

    // Read the incoming string until a newline character is received
    String command = Serial.readStringUntil('\n');

    // CRITICAL: Trim whitespace and invisible characters (like \r) from the end
    command.trim();

    // Execute command based on the string received
    if (command == "pneumo1") {
      digitalWrite(LED_BUILTIN, HIGH);
      digitalWrite(relayPin, HIGH);
      valveState = true;
      Serial.println("Command Received: Valve OPEN (Relay ON)");
    }
    else if (command == "pneumo0") {
      digitalWrite(LED_BUILTIN, LOW);
      digitalWrite(relayPin, LOW);
      valveState = false;
      Serial.println("Command Received: Valve CLOSED (Relay OFF)");
    }
    else if (command == "status") {
      Serial.print("Current Status: ");
      Serial.println(valveState ? "OPEN" : "CLOSED");
    }
    else if (command.length() > 0) {
      // Catch-all for unrecognized commands (ignores empty lines)
      Serial.print("Unknown command: ");
      Serial.println(command);
    }
  }
}
