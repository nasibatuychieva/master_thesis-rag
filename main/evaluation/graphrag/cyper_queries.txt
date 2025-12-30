1:
MATCH (e1:Entity)-[r1]-(e2:Entity)-[r2]-(e3:Entity)-[r3]-(e4:Entity)-[r4]-(e5:Entity)
WHERE e1.id in ["Dac", "DAC"]
and e1.entityType = "Peripheral"
and e2.entityType = "Product"
and e3.entityType = "Component"
and e3.id = "Rgb Led"
and e5.entityType = "Product"
and e5.id in [
  "GIGA R1 WiFi",
  "Braccio Carrier",
  "Mega 2560 Rev3",
  "Plug and Make Kit",
  "Due",
  "GIGA Display Shield",
  "DIN Celsius",
  "DIN Simul8",
  "Make Your UNO Kit",
  "Alvik",
  "Cloud Editor",
  "Node Red",
  "MKR IoT Carrier",
  "MKR WAN 1310",
  "MKR IoT Carrier Rev2",
  "MKR WiFi 1010",
  "MKR Vidor 4000",
  "Modulino Buzzer",
  "Modulino Thermo",
  "Nicla Sense ME",
  "Modulino Distance",
  "Modulino Pixels",
  "Nicla Vision",
  "Modulino Buttons",
  "Nicla Voice",
  "Nicla Sense Env",
  "Opta",
  "Opta Analog Expansion A0602",
  "Portenta Machine Control",
  "Opta Digital Expansion D1608E - D1608S",
  "WisGate Edge Pro",
  "Portenta Proto Kit ME",
  "Edge Control",
  "Stella",
  "Portenta Proto Kit VE",
  "WisGate Edge Lite 2",
  "Nano Connector Carrier",
  "Nano 33 IoT",
  "Nano 33 BLE",
  "Nano Every",
  "Nano RP2040 Connect",
  "Nano ESP32",
  "Nano 33 BLE Sense",
  "Nano 33 BLE Rev2",
  "Nano 33 BLE Sense Rev2",
  "Nano",
  "Nano Screw Terminal Adapter",
  "Nano Matter",
  "Nano R4"
]
RETURN distinct e1,r1,e2,r2,e3, r3, e4, r4, e5

