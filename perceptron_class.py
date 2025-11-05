#####################################################################
#  Filename:    perceptron_class.py
#  Versión:     1.0  (05-11-2025)
#  Autor:       Jose M Morales 
#  Web:         https://playbyte.es/electronica/20250122_percepduino
#  Licencia:    Creative Commons Share-Alike 4.0
#               https://creativecommons.org/licenses/by-nc-sa/4.0/
#
#  Descripción: Clase en Python que implementa un PERCEPTRÓN simple
#               con 'n' entradas analogicas, una salida digital ŷ 
#                           _________
#                           |       |
#                     x1 -->|w1     |       ⌈ 0, si  w1x1+w2x2 +... ≤ b 
#                     x2 -->|w2   _ |--> ŷ= |
#                     ..    |   _|  |       ⌊ 1, si  w1x1+w2x2 +... > b
#                     xn -->|wn     |
#                           |___w0__|
#                               |
#                               b
#####################################################################*/
# import numpy as np  # Requiere libreria numpy 


class Perceptron:
    """
    Perceptrón simple como clasificador lineal.
    
    Entradas: 
                        X: [x1, x2, ..., xn]
                        activación: 'step', 'relu', 'sigmoid', 'tanh'
                        lr: tasa de aprendizaje

    Parámetros internos:  w0 (bias), w1, w2, ..., wn
    """

    def __init__(self, n, activacion="step", lr=0.01):
        self.n = n + 1                     # el bias 'b' se considera como una entrada mas
        self.w = np.random.randn(self.n)   # asigna valores a los pesos 
        self.act = activacion              # función de activación del perceptron
        self.lr = lr

        # Historiales
        self.errores = []                  # lista de errores en predicciones (un epoch)
        self.lista_errores = []            # historico de errores (para graficar)
        self.lista_pesos = [self.w.copy()] # historico de pesos (para graficar)
        self.lista_loss = []               # valor promedio del error (loss) en cada época
        self.lista_accuracy = []           # porcentaje de aciertos del modelo en cada época
        self.score_  = 0.0                 # último valor calculado de la precisión (accuracy)
        self.epochs_ = 0
    # ==========================================================


    def suma_ponderada(self, xi):
        """
        Calcula la suma ponderada: b + w1x1 + w2x2 + ...
        """
        self.X = np.append(1, xi)  # X = [1, x1, x2, ...] vector de entradas aumentado
        return self.w.dot(self.X)
    # ==========================================================


    def activacion_func(self, z: float) -> float:
        """
        Aplica la función de activación.
        """
        if self.act == "step":
            # Funcion hard tresshold, step o umbral (default)
            return 1.0 if z >= 0 else 0.0

        elif self.act == "relu":
            # Funcion ReLu
            return max(0, z)

        elif self.act == "sigmoid":
            # Funcion logistica (sigmoide)
            return 1. / (1 + np.exp(-z))

        elif self.act == "tanh":
            # Funcion tangente hiperbolica
            return np.tanh(z)

        else:
            raise ValueError(f"Función de activación '{self.act}' no reconocida.")
    # ==========================================================
 

    def predict(self, xi):
        """
        Propagación hacia adelante: predice la etiqueta de una entrada xi.
        Método que predice la etiqueta para datos no vistos,
        en base a los datos de entrada y los pesos ajustados.
        """
        y_hat = self.activacion_func(self.suma_ponderada(xi))

        # Binariza la salida si es una función continua
        if self.act in ["sigmoid", "tanh"]:
            return 1.0 if y_hat >= 0.5 else 0.0
        return y_hat
    # ==========================================================


    def update_pesos(self, x_train, y_train):
        """
        Recibe UN dato de entrenamiento
        Actualiza los pesos solo si el error es no nulo
        """
        y_pred = self.predict(x_train) # la salida obtenida (predicha) por el perceptron
        error = y_train - y_pred       # diferencia entre la salida correcta y la obtenida

        if error == 0:   return 0      # dato correcto, pasa al siguiente dato

        # Actualización de pesos
        for i in range(self.n):
            # ajusta el valor de los pesos (de todos los xi donde la entrada sea !=0)
            self.w[i] += self.lr * error * self.X[i]  # X incluye X[0]=1

        self.lista_pesos.append(self.w.copy())
        return error
    # ==========================================================


    def fit(self, train_data, lr=0.05, max_epoch=2000, verbose=0, resumen=True):
        """ 
        El método fit se utiliza para entrenar el modelo  con los datos de entrenamiento, 
        ajustando los pesos en base a los datos de entrada y las etiquetas correspondientes.
        
        INPUT
        -----
        X : numpy 2D array. Cada fila corresponde a un ejemplo de entrenamiento.
        y : numpy 1D array. Etiqueta (0 ó 1) de cada ejemplo.
        
        OUTPUT
        ------
        self: El modelo entrenado.
        Retorna array de pesos wi tras el entrenamiento
        """

        self.lr = lr          # modifica valor por defecto 
        X = train_data[:, :2] # Extrae X=(x1,x2,...)
        y = train_data[:, 2]  # Extrae y=(y1,y2,...)

        epoca = 0
        n_err = 1
        self.errors_.clear()

        while epoca < max_epoch:
            self.errores.clear()
            epoca += 1       # cuenta numero de iteraciones
            n_err = 0        # numero de datos mal clasificados

            for dato in train_data:
                x_train = dato[:2]  # extrae caracteristicas [x1,x2,..]
                y_train = dato[2]   # extrae etiquetas [y] (salida esperada)
                
                # actualiza pesos (si hay error)
                err = self.update_pesos(x_train, y_train)
                self.errores.append(err)  # añade resultado a la lista de errores
                if err != 0:  n_err += 1  # dato mal clasificado 

            # Calcula métricas
            epoch_score = self.score(X, y)
            self.lista_accuracy.append(epoch_score)
            self.errors_.append(n_err)
            mean_loss = np.mean([abs(e) for e in self.errores])
            self.lista_loss.append(mean_loss)

            if verbose:
                print(f"Época {epoca:3d} | Errores: {n_err:3d} | "
                      f"Score: {epoch_score:.3f} | Loss: {mean_loss:.3f}")

            if n_err == 0:
                if verbose:
                    print(f"Entrenamiento completado en {epoca} iteraciones ✅")
                break

        self.epochs_ = epoca

        if n_err != 0 and verbose:
            print(f"Entrenamiento NO convergió tras {max_epoch} épocas.")
            print(f"Revisa si los datos son linealmente separables o ajusta lr={self.lr}")

        if resumen:
            self.mostrar_resumen(X, y)

        return self.lista_pesos
    # ==========================================================


    @staticmethod
    def loss(y_pred, y_true):
        """
        Error cuadrático medio.
        """
        return np.mean((y_pred - y_true) ** 2)

    @staticmethod
    def accuracy(y_pred, y_true):
        """
        Exactitud del modelo.
        """
        correct = np.sum(y_pred == y_true)
        return correct / len(y_true)
    # ==========================================================


    def score(self, X, y):
        """
        Calcula la proporción de aciertos del modelo.
        """
        predictions = np.array([self.predict(xi) for xi in X])
        self.score_ = round(self.accuracy(predictions, y), 4)
        return self.score_
    # ==========================================================


    def mostrar_resumen(self, X, y):
        """
        Muestra un resumen del estado del modelo tras el entrenamiento.
        """
        print("\n================ RESULTADOS DEL ENTRENAMIENTO ================")
        print(f"  Épocas ejecutadas: {self.epochs_}")
        print(f"  Tasa de aprendizaje (lr): {self.lr}")
        print(f"  Precisión final (score): {self.score_:.4f}")
        print(f"  Último error promedio:   {self.lista_loss[-1]:.4f}")
        print(f"  Pesos finales: {np.round(self.w, 4)}")
        print("==============================================================\n")        
######################################################################