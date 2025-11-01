/* ####################  Percepduino  ####################

*  Filename:    Percepduino_Ej2.ino

*  Versión:     1.0  (31-10-2025)

*  Descripción: Perceptron simple de dos entradas,

*               Entrenamiento con puntos introducidos manualmente

*               Numero de puntos variable

*  Emulación:   https://wokwi.com/projects/446333103632678913

*  Autor:       Jose M Morales 

*  Web:         https://playbyte.es/articulos/electronica/20250122_perceptron/percepduino.html

*  Licencia:    Creative Commons Share-Alike 4.0

*               https://creativecommons.org/licenses/by-nc-sa/4.0/

* ######################################################*/


#include <LiquidCrystal_I2C.h>

#define I2C_ADDR 0x27

#define LCD_COLUMNS 20

#define LCD_LINES 4


LiquidCrystal_I2C lcd(I2C_ADDR, LCD_COLUMNS, LCD_LINES);

// Entradas analógicas

#define SENSOR_POT1 A1

#define SENSOR_POT2 A2


// Salidas

const int pin_LED_red = 2;

const int pin_LED_green = 3;

const int pin_LED_yellow = 4;

const int pin_class1 = 9;

const int pin_class0 = 10;


// ====== PARÁMETROS ======

const float lr = 0.1;

const float w1_i = 0.4;

const float w2_i = 0.4;

const float b_i  = -0.6;

const int max_epoch = 100;

const int max_samples = 20; // ahora soporta hasta 20 puntos


// ====== DATOS ======

float X_train[max_samples][2] = {0};

int y_train[max_samples] = {0};

int N_samples = 0;

int N_objetivo = 4;

bool perceptron_entrenado = false;


// ====== CLASE PERCEPTRON ======

class Perceptron {

  private:

    float w1, w2, b;

    float lr;


  public:

    int total_epocas;

    int total_errores;


    Perceptron(float learning_rate = 0.1, 
        float w1_i = NAN, 
        float w2_i = NAN, 
        float b_i = NAN) {

      lr = learning_rate;

      w1 = isnan(w1_i) ? (random(-5000, 5001) / 10000.0) : w1_i;

      w2 = isnan(w2_i) ? (random(-5000, 5001) / 10000.0) : w2_i;

      b  = isnan(b_i)  ? (random(-5000, 5001) / 10000.0) : b_i;

      total_epocas = 0;

      total_errores = 0;

    }


    int predict(float x1, float x2) {

      float z = (x1*w1 + x2*w2) + b;

      return z >= 0 ? 1 : 0;

    }


    void train(float X[][2], int y[], int N, int max_epoch) {

      for (int epoch = 0; epoch < max_epoch; epoch++) {

        total_errores = 0;

        for (int n = 0; n < N; n++) {

          int y_hat = predict(X[n][0], X[n][1]);

          int error = y[n] - y_hat;

          if (error != 0) {

            total_errores++;

            w1 += lr * error * X[n][0];

            w2 += lr * error * X[n][1];

            b  += lr * error;

          }

        }

        total_epocas = epoch;

        if (total_errores == 0) break;

      }

    }


    float get_w1() { return w1; }

    float get_w2() { return w2; }

    float get_b()  { return b;  }

    int get_epoch() { return total_epocas; }

    int get_errors() { return total_errores; }

};


// ====== INSTANCIA GLOBAL ======

Perceptron mi_percepduino(lr, w1_i, w2_i, b_i);


// ====== SETUP ======

void setup() {

  pinMode(pin_class0, INPUT_PULLUP);

  pinMode(pin_class1, INPUT_PULLUP);

  pinMode(pin_LED_red, OUTPUT);

  pinMode(pin_LED_green, OUTPUT);

  pinMode(pin_LED_yellow, OUTPUT);

  lcd.init();

  lcd.backlight();


  Serial.begin(9600);

  Serial.println("======== Percepduino v1.1 ========");

  Serial.println("Perceptrón simple de dos entradas.");

  Serial.println("Introduce el número de puntos a entrenar (máx 20):");


  while (Serial.available() == 0);

  N_objetivo = Serial.parseInt();

  if (N_objetivo < 2) N_objetivo = 2;

  if (N_objetivo > max_samples) N_objetivo = max_samples;

  Serial.print("Número de puntos a introducir: ");

  Serial.println(N_objetivo);


  lcd.clear();

  lcd.setCursor(0, 0);

  lcd.print("Percepduino listo!");

  lcd.setCursor(0, 2);

  lcd.print("Puntos: ");

  lcd.print(N_objetivo);

  delay(2000);

  lcd.clear();

}


// ====== LOOP PRINCIPAL ======

void loop() {

  if (!perceptron_entrenado) {

    modoInput();

  } else {

    modoPrediccion();

  }

}



// ====== FUNCIONES ======

void modoInput() {

  float x1 = 1 - analogRead(SENSOR_POT1) / 1023.0;

  float x2 = 1 - analogRead(SENSOR_POT2) / 1023.0;


  mostrarDatosLCD(x1, x2, "<<< Modo INPUT >>>");

  digitalWrite(pin_LED_yellow, HIGH);

  lcd.setCursor(0, 3);

  lcd.print("Punto ");

  lcd.print(N_samples + 1);

  lcd.print(": Clase(0/1)? ");


  int y_real = 2;

  if (digitalRead(pin_class0) == LOW) y_real = 0;

  if (digitalRead(pin_class1) == LOW) y_real = 1;


  if (y_real < 2 && N_samples < N_objetivo) {

    X_train[N_samples][0] = x1;

    X_train[N_samples][1] = x2;

    y_train[N_samples] = y_real;

    N_samples++;


    lcd.setCursor(0, 3);

    lcd.print(">>>>>>> y=");

    lcd.print(y_real);

    lcd.print(" <<<<<<<<");

    Serial.println("Punto " + String(N_samples) + ": x1=" + String(x1, 2) + ", x2=" + String(x2, 2) + " -> y=" + String(y_real));

    delay(1000);

  }


  if (N_samples == N_objetivo) {

    mostrarDatosSerial();

    digitalWrite(pin_LED_yellow, LOW);

    entrenar();

  }

}


void mostrarDatosSerial() {

  Serial.println("\\n=== Datos introducidos ===");

  for (int i = 0; i < N_samples; i++) {

    Serial.print("Punto ");

    Serial.print(i + 1);

    Serial.print(": [");

    Serial.print(X_train[i][0], 2);

    Serial.print(", ");

    Serial.print(X_train[i][1], 2);

    Serial.print("] -> y=");

    Serial.println(y_train[i]);

  }

  Serial.println("===========================");

}


void entrenar() {

  lcd.clear();

  lcd.print(" Entrenando...");

  mi_percepduino.train(X_train, y_train, N_samples, max_epoch);


  Serial.println("\\n=== RESULTADO ENTRENAMIENTO ===");

  Serial.print("Errores finales: "); Serial.println(mi_percepduino.get_errors());

  Serial.print("Epocas totales: "); Serial.println(mi_percepduino.get_epoch());

  Serial.print("w1="); Serial.println(mi_percepduino.get_w1(), 4);

  Serial.print("w2="); Serial.println(mi_percepduino.get_w2(), 4);

  Serial.print("b ="); Serial.println(mi_percepduino.get_b(), 4);

  Serial.println("===============================");


  if (mi_percepduino.get_errors() == 0) {

    perceptron_entrenado = true;

    lcd.clear();

    lcd.print("< Entrenamiento OK >");

    delay(2000);

  } else {

    lcd.clear();

    lcd.print("Fallo entrenamiento");

    lcd.setCursor(0, 1);

    lcd.print("Repite los puntos");

    Serial.println("⚠️ Entrenamiento no completado. Repite los puntos.");

    delay(2500);

    lcd.clear();

    N_samples = 0;

    perceptron_entrenado = false;

  }

}


void modoPrediccion() {

  float x1 = 1 - analogRead(SENSOR_POT1) / 1023.0;

  float x2 = 1 - analogRead(SENSOR_POT2) / 1023.0;

  int y_hat = mi_percepduino.predict(x1, x2);


  digitalWrite(pin_LED_red, y_hat);

  digitalWrite(pin_LED_green, !y_hat);


  mostrarDatosLCD(x1, x2, "< Modo PREDICCION > ");

  lcd.setCursor(0, 3);

  lcd.print("Salida: y_hat=" + String(y_hat));

  delay(200);

}


void mostrarDatosLCD(float x1, float x2, String mode) {

  lcd.setCursor(0, 0);

  lcd.print(mode);

  lcd.setCursor(0, 1);

  lcd.print("x1= " + String(x1, 2));

  lcd.setCursor(0, 2);

  lcd.print("x2= " + String(x2, 2));

}
