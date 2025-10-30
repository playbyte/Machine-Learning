/* #################### -- PERCEPTRON --  ####################
*  Filename:    perceptron_class.h
*  Versión:     1.0  (26-10-2025)
*  Autor:       Jose M Morales 
*  Web:         https://playbyte.es/electronica/20250122_percepduino
*  Licencia:    Creative Commons Share-Alike 4.0
                https://creativecommons.org/licenses/by-nc-sa/4.0/

*  Descripción: Clase que implementa un PERCEPTRÓN simple para Arduino
                Dos entradas x1 y x2, analogicas, una salida ŷ digital            
                           _________
                           |       |
                     x1 -->|w1   _ |       ⌈ 0, si  w1x1 + w2x2 ≤ b 
                           |   _|  |--> ŷ= |
                     x2 -->|w2     |       ⌊ 1, si  w1x1 + w2x2 > b
                           |___w0__|                     
                               |
                               b
* #########################################################*/



class Perceptron {

  private:
      float w1, w2, b;   // Pesos y bias
      float lr;          // Tasa de aprendizaje
      int total_epocas;  // Numero de epocas ejecutadas
      int total_errores; // Cantidad de puntos mal clasificados

  public:  
    // ***** Constructor ************************************
    Perceptron( float learning_rate = 0.1,
                float w1_i = NAN,
                float w2_i = NAN,
                float b_i  = NAN) {

        lr = learning_rate;
        // Si no se pasan pesos, inicializa aleatoriamente en [-0.5, 0.5]
        w1 = isnan(w1_i) ? (random(-5000, 5001) / 10000.0) : w1_i;
        w2 = isnan(w2_i) ? (random(-5000, 5001) / 10000.0) : w2_i;
        b  = isnan(b_i)  ? (random(-5000, 5001) / 10000.0) : b_i;

        total_epocas  = 0;
        total_errores = 0;
      }//******************************************************


    // ***** PREDICCION **************
    int predict(float x1, float x2) { 
        
        float z = (x1*w1 + x2*w2) + b; // función de activación
        return z >= 0 ? 1 : 0;         // función step 
    }

    // ***** ENTRENAMIENTO ************
    void train(float X[][2], int y[], int N_samples, int max_epoch) {
        
        int error;  // [-1,1] (error=0 => punto bien clasificado)
  
        for (int epoch=0; epoch<max_epoch; epoch++) {

          total_errores =0; 
          // (1) Recorre todos los ejemplos ---------------
          for (int n=0; n<N_samples; n++) {

              // (2) Calcula la predicción y_hat
              int y_hat = predict(X[n][0], X[n][1]);
              // (3) Calcula el error
              error = y[n] - y_hat;
   
              if (error != 0) { // actualiza pesos

                total_errores++;
                // algoritmo aprendizaje del perceptrón clásico. 
                w1 += lr * error * X[n][0];
                w2 += lr * error * X[n][1];
                b  += lr * error;            
              }
          }// FIN bucle con todos los puntos -------------

        total_epocas = epoch;
  
        if (total_errores == 0) {
            Serial.println(F("**** ENTRENAMIENTO COMPLETADO ****"));
            print_status(epoch);
            return;
        }
        // Epoca completada (con errores)
        Serial.print(F("Época "));
        Serial.print(epoch);
        Serial.print(F(" | Errores: "));
        Serial.println(total_errores);

        }// FIN bucle de epoch ****************************

      Serial.println(F("**** ENTRENAMIENTO NO COMPLETADO ****"));
      print_status(max_epoch); // alcanzado el limite de epocas

    }// FIN funcion train() **************************************


    // ***** Muestra estado actual *******************************
    void print_status(int epoch) {
        Serial.print(F("Épocas: "));
        Serial.print(epoch);
        Serial.print(F(" | Errores: "));
        Serial.println(total_errores);
        Serial.print(F(" w1="));
        Serial.print(w1, 4);
        Serial.print(F(" w2="));
        Serial.print(w2, 4);
        Serial.print(F(" b="));
        Serial.println(b, 4);
    }

    // 'getters', permiten consultar parámetros desde el programa principal
    float get_lr() const      { return lr; }
    float get_w1() const      { return w1; }
    float get_w2() const      { return w2; }
    float get_b() const       { return b; }
    int   get_epoch() const   { return total_epocas; }   
    int   get_errors() const  { return total_errores; }
};