/* ####################  Percepduino  ####################
*  Filename:    Percepduino.ino
*  Versión:     1.0  (28-10-2025)
*  Descripción: Perceptron simple de dos entradas,
*               Entrenamiento con puntos introducidos manualmente
*
*  Emulación:   https://wokwi.com/projects/422918588393226241
*             
*  Autor:       Jose M Morales 
*  Web:         https://playbyte.es/articulos/electronica/20250122_perceptron/percepduino.html
*  Licencia:    Creative Commons Share-Alike 4.0
*               https://creativecommons.org/licenses/by-nc-sa/4.0/
* ######################################################*/


/**********  Pantalla LCD20x4, protocolo I2C *********/
#include <LiquidCrystal_I2C.h>

#define I2C_ADDR    0x27
#define LCD_COLUMNS 20
#define LCD_LINES   4
LiquidCrystal_I2C lcd(I2C_ADDR, LCD_COLUMNS, LCD_LINES);


/*****************  Definicion pines *****************/
// Entradas analógicas
#define SENSOR_POT1 A1  // potenciómetro 1
#define SENSOR_POT2 A2  // potenciómetro 2
// Salidas
const int pin_LED_red   = 2; // LED rojo
const int pin_LED_green = 3; // LED verde
const int pin_LED_yellow= 4; // LED amarillo
//const int pin_MODE      = 8; // Selecciona Modo
const int pin_class1    = 9; // clase1
const int pin_class0    =10; // clase0


/************  PARAMETROS DE ENTRENAMIENTO ************/
const float     lr = 0.1; // learning rate
const float   w1_i = 0.4; // peso inicial
const float   w2_i = 0.4;
const float    b_i =-0.6;
const int max_epoch= 100; // limite de iteraciones
const int max_samples= 4; // limite de puntos a clasificar


/**************  DATOS DE ENTRENAMIENTO **************/
float X_train[max_samples][2] = {0}; // entradas [x1,x2] 
int   y_train[max_samples] = {0};    // salida   y_hat
int   N_samples = 0;
bool  perceptron_entrenado = false;


/******* Ejemplos (puertas logicas) *******/

float data_AND[4][3]= {       // puerta AND
                      {0,0,0},
                      {0,1,0},
                      {1,0,0},
                      {1,1,1}  
                     };
float data_OR[4][3]= {        // puerta OR
                      {0,0,0},
                      {0,1,1},
                      {1,0,1},
                      {1,1,1}  
                     };
float data_NAND[4][3]={       // puerta NAND
                      {0,0,1},
                      {0,1,1},
                      {1,0,1},
                      {1,1,0}  
                     };
float data_NOR[4][3]= {        // puerta NOR
                      {0,0,1},
                      {0,1,0},
                      {1,0,0},
                      {1,1,0}  
                     };                     
float data_XOR[4][3]= {        // puerta XOR (sin solucion)
                      {0,0,0},
                      {0,1,1},
                      {1,0,1},
                      {1,1,0}  
                     };

/*****************************************************/



//###############  Clase PERCEPTRON ################
class Perceptron {

  private:
      float w1, w2, b;  // Pesos y bias
      float lr;         // Tasa de aprendizaje

  public:
      int total_epocas;  // contador de epocas
      int total_errores; // cuenta puntos mal clasificados

    // ***** Constructor *****
    Perceptron( float learning_rate = 0.1,
                float w1_i = NAN,
                float w2_i = NAN,
                float b_i  = NAN) {
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
      int error;
      for (int epoch = 0; epoch < max_epoch; epoch++) {
        total_errores = 0;
        for (int n = 0; n < N; n++) {
          int y_hat = predict(X[n][0], X[n][1]);
          error = y[n] - y_hat;
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

    float get_w1(){return w1;}
    float get_w2(){return w2;}
    float get_b(){return b;}
    int get_epoch(){return total_epocas;}
    int get_errors(){return total_errores;}
};

// Inicializa perceptron (instancia global)
Perceptron mi_percepduino(lr, w1_i, w2_i, b_i); 


// #########################################################
//                           SETUP
// #########################################################
void setup() {
//  pinMode(pin_MODE, INPUT_PULLUP);
  pinMode(pin_class0, INPUT_PULLUP);
  pinMode(pin_class1, INPUT_PULLUP);
  pinMode(pin_LED_red, OUTPUT);
  pinMode(pin_LED_green, OUTPUT);
  pinMode(pin_LED_yellow, OUTPUT);

  lcd.init();
  lcd.backlight();
  lcd.setCursor(0, 0);
  lcd.print("  PERCEPDUINO v1.0  ");
  lcd.setCursor(0, 2);
  lcd.print(" Esperando datos... ");

  Serial.begin(9600);
  Serial.println(F("  ========= Percepduino v1.0 ========="));
  Serial.println(F("Neurona básica de 2 entradas que entrena con puntos introducidos manualmente usando el algoritmo del perceptron.\n"));
  Serial.println(F("1- Introduce punto (x1,x2) mediante el potencimetro"));
  Serial.println(F("2- Introduce el valor de la salida 'y_real' (clases) usando los botones")); 
  Serial.println(F("3- Tras introducir 4 puntos realiza el entrenamiento")); 
  Serial.println(F("4- Clasifica los puntos que se introduzcan")); 
   
  Serial.println(F("  ==================================\n"));
  delay(2000);
  lcd.clear();
}

// ==========================================================
//                     LOOP PRINCIPAL
// ==========================================================
void loop() {

  if (!perceptron_entrenado) {
    modoInput();      // introduce puntos y entrena

  } else {
    modoPrediccion(); // ya entrenado → predicción libre
  }
}

// ==========================================================
//                  FUNCIONES AUXILIARES
// ==========================================================
void modoInput() {

  float x1 = 1 - analogRead(SENSOR_POT1) / 1023.0;
  float x2 = 1 - analogRead(SENSOR_POT2) / 1023.0;

  mostrarDatosLCD(x1,x2, "<<< Modo INPUT >>>");
  digitalWrite(pin_LED_yellow, HIGH);
  lcd.setCursor(0, 3);
  lcd.print("Punto ");
  lcd.print(N_samples + 1);
  lcd.print(": Clase(0/1)? ");

  int y_real = 2;
  if (digitalRead(pin_class0) == LOW) y_real = 0;
  if (digitalRead(pin_class1) == LOW) y_real = 1;

  if (y_real < 2 && N_samples < max_samples) {
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

  if (N_samples == 4) {
    // FIN entrada de datos
    mostrarDatosSerial();
    digitalWrite(pin_LED_yellow, LOW);
    entrenar();
  }
}

void mostrarDatosSerial() {
  Serial.println("\n=== Datos introducidos ===");
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

  // --- Resultado por Serial ---
  Serial.println("\n=== RESULTADO ENTRENAMIENTO ===");
  Serial.print("Errores finales: "); Serial.println(mi_percepduino.get_errors());
  Serial.print("Epocas totales: "); Serial.println(mi_percepduino.get_epoch());
  Serial.print("w1="); Serial.println(mi_percepduino.get_w1(), 4);
  Serial.print("w2="); Serial.println(mi_percepduino.get_w2(), 4);
  Serial.print("b ="); Serial.println(mi_percepduino.get_b(), 4);
  Serial.println("===============================");


  if (mi_percepduino.get_errors() == 0) {
  // entrenamiento completado    
    perceptron_entrenado = true;
    lcd.clear();
    lcd.print("< Entrenamiento OK >");
    delay(2000);
    verificarPuertaLogica();

  } else {
    // entrenamiento fallido
    lcd.clear();
    lcd.print("Fallo entrenamiento");
    lcd.setCursor(0, 1);
    lcd.print("Repite los 4 puntos");
    Serial.println("⚠️ Entrenamiento no completado.");
    Serial.println("Introduce nuevos 4 puntos.");
  
    delay(2500);
    lcd.clear();    
    N_samples = 0;
    perceptron_entrenado = false;
  }
}


void verificarPuertaLogica() {

  int match_AND = 0;
  int match_OR  = 0;

  for (int i = 0; i < 4; i++) {
    // Evaluar AND
    int y_hat_AND = mi_percepduino.predict(data_AND[i][0], data_AND[i][1]);
    if (y_hat_AND == (int)data_AND[i][2]) match_AND++;

    // Evaluar OR
    int y_hat_OR = mi_percepduino.predict(data_OR[i][0], data_OR[i][1]);
    if (y_hat_OR == (int)data_OR[i][2]) match_OR++;
  }

  lcd.clear();
  if (match_AND == 4) {
    lcd.print("Compatible con AND");
    Serial.println("✅ Compatible con la puerta lógica AND");
  } 
  else if (match_OR == 4) {
    lcd.print("Compatible con OR");
    Serial.println("✅ Compatible con la puerta lógica OR");
  } 
  else {
  //  lcd.print("Sin coincidencias");
    Serial.println("ℹ️ Sin coincidencias con AND ni OR");
  }
  delay(2500);
  lcd.clear();
}


void modoPrediccion() {
  float x1 = 1 - analogRead(SENSOR_POT1) / 1023.0;
  float x2 = 1 - analogRead(SENSOR_POT2) / 1023.0;
  int y_hat = mi_percepduino.predict(x1, x2);

  digitalWrite(pin_LED_red, y_hat);
  digitalWrite(pin_LED_green, !y_hat);

  mostrarDatosLCD(x1,x2, "< Modo PREDICCION > ");
  lcd.setCursor(0, 3);
  lcd.print("Salida: y_hat=" + String(y_hat));
  delay(200);
}


void mostrarDatosLCD(float x1, float x2, String mode)  {

  lcd.setCursor(0, 0);
  lcd.print(mode);
  lcd.setCursor(0, 1);
  lcd.print("x1= " + String(x1, 2));
  lcd.setCursor(0, 2);
  lcd.print("x2= " + String(x2, 2));
} 