/*
 * Created by ArduinoGetStarted.com
 *
 * This example code is in the public domain
 *
 * Tutorial page: https://arduinogetstarted.com/tutorials/arduino-neopixel-led-strip
 */

#include <Adafruit_NeoPixel.h>
#ifdef __AVR__
#include <avr/power.h> // Required for 16 MHz Adafruit Trinket
#endif

#define PIN_NEO_PIXEL  2   // Arduino pin that connects to NeoPixel
#define NUM_PIXELS     60  // The number of LEDs (pixels) on NeoPixel

Adafruit_NeoPixel NeoPixel(NUM_PIXELS, PIN_NEO_PIXEL, NEO_GRB + NEO_KHZ800);

void setup() {
  pinMode(PIN_NEO_PIXEL, OUTPUT);
  NeoPixel.begin(); // INITIALIZE NeoPixel strip object (REQUIRED)
  Serial.begin(9600);

  for (int i = 0; i < 8; i++) {
    pinMode(3+i, INPUT_PULLUP);
  }
}

void loop() {
  NeoPixel.clear(); // set all pixel colors to 'off'. It only takes effect if pixels.show() is called

  // turn pixels to green one by one with delay between each pixel
  for (int pixel = 0; pixel < NUM_PIXELS; pixel++) { // for each pixel
    NeoPixel.setPixelColor(pixel, NeoPixel.Color(255, 255, 255)); // it only takes effect if pixels.show() is called

  }
  NeoPixel.show();   // send the updated pixel colors to the NeoPixel hardware. 

  Serial.print(",");
  for (int i = 0; i < 8; i++) {
    if( digitalRead(i+3)) {
      Serial.print(".");
    } else {
      Serial.print("#");
    }
    Serial.print(",");
  }
  Serial.println();

  delay(50);
}
