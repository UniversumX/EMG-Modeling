#include <Arduino.h>
#include <math.h>

#define EMG_PIN 1

// Number of samples per window
#define SAMPLE_SIZE 1000

float meanAbsoluteValue(float *signal, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += abs(signal[i]);
    }
    return sum / size;
}

float waveformLength(float *signal, int size) {
    float sum = 0;
    for (int i = 1; i < size; i++) {
        sum += abs(signal[i] - signal[i - 1]);
    }
    return sum;
}

int zeroCrossing(float *signal, int size) {
    int count = 0;
    for (int i = 1; i < size; i++) {
        if ((signal[i] > 0 && signal[i - 1] < 0) || (signal[i] < 0 && signal[i - 1] > 0)) {
            count++;
        }
    }
    return count;
}

int slopeSignChange(float *signal, int size) {
    int count = 0;
    for (int i = 2; i < size; i++) {
        if ((signal[i] - signal[i - 1]) * (signal[i - 1] - signal[i - 2]) < 0) {
            count++;
        }
    }
    return count;
}

float rootMeanSquare(float *signal, int size) {
    float sum = 0;
    for (int i = 0; i < size; i++) {
        sum += signal[i] * signal[i];
    }
    return sqrt(sum / size);
}

void setup() {
    Serial.begin(115200);
    pinMode(EMG_PIN, INPUT);
}

void loop() {
    float emg_signal[SAMPLE_SIZE];

    for (int i = 0; i < SAMPLE_SIZE; i++) {
        emg_signal[i] = analogRead(EMG_PIN);  
        delayMicroseconds(100);  // Sampling Rate

        // Number of samples collected / second = 1,000,000 / delayTime - here sampling_rate = 10kHz 
        // since sampling rate = 10kHz, no of samples / inference window = 1000, we can run 10 cycles of feature extraction / second
        // these are arbitrary values for now, and need to be modified considering real time analysis and accounting for inference lag
    }

    // all of the features below are O(n) as currently implemented (with no libraries / optimization)
    // this will also account for time loss
    // will explore alternatives to this

    float mav = meanAbsoluteValue(emg_signal, SAMPLE_SIZE);
    float wl = waveformLength(emg_signal, SAMPLE_SIZE);
    int zc = zeroCrossing(emg_signal, SAMPLE_SIZE);
    int ssc = slopeSignChange(emg_signal, SAMPLE_SIZE);
    float rms = rootMeanSquare(emg_signal, SAMPLE_SIZE);

    Serial.print("MAV: "); Serial.print(mav);
    Serial.print(", WL: "); Serial.print(wl);
    Serial.print(", ZC: "); Serial.print(zc);
    Serial.print(", SSC: "); Serial.print(ssc);
    Serial.print(", RMS: "); Serial.println(rms);

    delay(500);  // arbitrary delay - needs to be determined empirically
}
