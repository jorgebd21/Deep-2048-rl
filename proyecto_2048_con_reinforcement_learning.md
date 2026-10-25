# Propuesta de Proyecto: Agente Autónomo para 2048 mediante Deep Reinforcement Learning y Aceleración por GPU

---

## 1. Resumen Ejecutivo

Este documento define la arquitectura, diseño técnico y plan de implementación para el desarrollo de un agente de Aprendizaje por Refuerzo Profundo (*Deep Reinforcement Learning* o DRL) capaz de dominar el juego de lógica combinatoria **2048**.

El objetivo principal es concebir un sistema que aprenda de forma autónoma estrategias óptimas de resolución (incluida la consolidación de fichas de alto valor en esquinas y la gestión del espacio en cascada) partiendo únicamente de la dinámica de recompensa intrínseca del juego, prescindiendo de reglas heurísticas programadas a mano.

El proyecto se estructura para aprovechar al máximo las capacidades de aceleración de hardware por GPU (arquitectura NVIDIA Ampere / RTX 3060 local e instancias AWS EC2 con Tesla T4), integrando computación matricial intensiva con PyTorch y paralelización del entorno.

---

## 2. Definición del Problema y Justificación Técnica

### 2.1. Dinámica del Entorno 2048
2048 es un juego estocástico de información perfecta con un espacio de estados acotado pero combinatoriamente masivo:
* **Espacio de Estados ($S$):** Cuadrícula de $4 \times 4$ casillas cuyos valores son potencias de 2 ($2^1$ a $2^{16}$) o 0 (casilla vacía).
* **Espacio de Acciones ($A$):** Cuatro acciones discretas: $\mathcal{A} \in \{\text{Arriba}, \text{Abajo}, \text{Izquierda}, \text{Derecha}\}$.
* **Transición ($T$):** Tras ejecutar una acción legal, todas las fichas se desplazan hacia el borde correspondiente colapsando parejas adyacentes iguales. Posteriormente, el entorno genera de forma aleatoria una nueva ficha (un 2 con probabilidad $p=0.9$, o un 4 con $p=0.1$) en una casilla desocupada.

### 2.2. Por qué construir un motor headless (sin GUI)
El entrenamiento de agentes mediante métodos basados en valor o gradiente de políticas requiere cientos de miles de partidas para estabilizarse. Ejecutar el agente interactuando contra una aplicación web, emuladores gráficos o capturas de pantalla (visión por computador vía OpenCV/Selenium) limitaría la tasa de simulación a 30-60 fotogramas por segundo, prolongando el entrenamiento durante semanas.

**Decisión técnica:** Se implementará un simulador nativo en Python/NumPy (compatible con la API estándar de *Gymnasium*), operando exclusivamente sobre tensores en memoria RAM/VRAM. Esto permite alcanzar órdenes de magnitud superiores en la tasa de pasos por segundo ($> 20.000\text{ steps/s}$).

---

## 3. Arquitectura del Sistema

```
+--------------------------------------------------------------------+
|                         BUCLE DE CONTROL                           |
|                                                                    |
|   +-----------------------+              +---------------------+   |
|   |                       |  Acción (a)  |                     |   |
|   |  Agente DRL (PyTorch) | -----------> |   Entorno 2048      |   |
|   |  - Red DQN / Dueling  |              |   (Gymnasium /      |   |
|   |  - Epsilon-Greedy     | <----------- |    Matriz 4x4)      |   |
|   |                       | Estado s', r |                     |   |
|   +-----------------------+              +---------------------+   |
|              |                                                     |
|              v (s, a, r, s', done)                                 |
|   +-----------------------+                                        |
|   | Replay Buffer         |                                        |
|   | (Muestreo Mini-Batch) |                                        |
|   +-----------------------+                                        |
|              |                                                     |
|              v Optimización por gradiente (Adam + MSE / Huber)     |
|   +-----------------------+                                        |
|   | GPU Acceleration      | (NVIDIA Ampere Tensor Cores / TF32)    |
|   +-----------------------+                                        |
+--------------------------------------------------------------------+
```

### 3.1. Representación del Estado (Input Representation)
Tratar la matriz $4 \times 4$ como un array de números escalares brutos ($[0, 2, 4, \dots, 2048]$) distorsiona el aprendizaje porque la escala numérica crece exponencialmente, dificultando la ponderación lineal en capas densas.

Se implementa una codificación en **canales One-Hot binarios**:
* Dimensiones del tensor de entrada: $\mathbf{X} \in \mathbb{R}^{B \times 16 \times 4 \times 4}$
* Donde cada uno de los 16 planos binarios representa si una celda contiene un valor específico:
  $$\text{Canal } k \implies \text{casilla con valor } 2^{k} \quad (k \in [1, 16] \text{ y canal } 0 \text{ para casilla vacía})$$
* **Ventaja:** Permite aplicar capas convolucionales bidimensionales (`Conv2d`) para que el modelo identifique relaciones topológicas locales entre fichas contiguas (monotonía y adyacencia).

### 3.2. Modelo Neuronal (Dueling Deep Q-Network)
Para mitigar la sobreestimación de valores de acción intrínseca a Q-Learning tradicional, se adopta una arquitectura **Dueling DQN**:

1. **Tronco Convolucional Común:**
   * Capas convolucionales con filtros pequeños ($2 \times 2$ y $1 \times 1$) y activación LeakyReLU/ReLU para extraer patrones geométricos en filas y columnas.
2. **Bifurcación Dueling:**
   * **Rama de Valor de Estado $V(s)$:** Capa densa que proyecta a un único escalar, estimando lo favorable que es la configuración actual del tablero.
   * **Rama de Ventaja $A(s, a)$:** Capa densa que proyecta a 4 salidas, estimando el beneficio relativo de cada movimiento respecto a los demás.
3. **Reconstrucción del valor Q:**
   $$Q(s, a; \theta, \alpha, \beta) = V(s; \theta, \beta) + \left( A(s, a; \theta, \alpha) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a'; \theta, \alpha) \right)$$

---

## 4. Función de Recompensa y Estabilidad Matemática

Para cumplir la premisa de no inducir reglas heurísticas artificiales ("no programar la esquina a mano"), la recompensa se deriva estrictamente de las consecuencias directas del juego:

### 4.1. Escala Logarítmica de Fusión
Dado que los puntos otorgados por fusión varían entre $+4$ y $+2048$, se emplea una normalización logarítmica para estabilizar el gradiente:
$$r_{\text{fusión}} = \begin{cases} 
\log_2(\text{puntos obtenidos}) & \text{si hay fusión} \\
0 & \text{si el movimiento no fusiona fichas}
\end{cases}$$

### 4.2. Filtrado de Acciones Inválidas
Para evitar que el agente se degrade ejecutando movimientos que no modifican el estado (acciones ilegales):
* El entorno devuelve una máscara booleana de acciones legales.
* Se aplica una penalización negativa fija ($r_{\text{inválida}} = -10$) o enmascaramiento directo de logits (*action masking*), obligando a la política a elegir únicamente movimientos que generen cambios de estado.

### 4.3. Emergencia Natural de la Heurística de Esquina
Al maximizar el retorno acumulado a largo plazo con un factor de descuento $\gamma \in [0.95, 0.99]$, el agente deduce que colocar la ficha mayor en un vértice minimiza el riesgo de fragmentación del tablero, maximizando la probabilidad de supervivencia y la acumulación sostenida de puntos.

---

## 5. Infraestructura y Estrategia de Cómputo

### 5.1. Comparativa de Entornos

| Parámetro | AWS EC2 (Tesla T4) | Entorno Local (RTX 3060 Desktop) |
| :--- | :--- | :--- |
| **Arquitectura** | Turing (`sm_75`) | Ampere (`sm_86`) |
| **VRAM** | 16 GB GDDR6 | 12 GB GDDR6 |
| **Unidades Matriciales** | 320 Tensor Cores (Gen 2) | 112 Tensor Cores (Gen 3) |
| **Formatos Acelerados** | FP16, INT8 | TF32 (nativo), BF16, FP16, INT8 |
| **Uso Principal** | Prácticas guiadas del curso (DLI) | Entrenamiento intensivo continuo sin coste |

### 5.2. Pipeline de Aceleración Local (PyTorch CUDA)
1. **Activación de TensorFloat-32 (TF32):**
   ```python
   torch.backends.cuda.matmul.allow_tf32 = True
   torch.backends.cudnn.allow_tf32 = True
   ```
2. **Buffer de Replay Vectorizado:** Mantenimiento de experiencias en tensores preasignados en CPU/GPU para minimizar la latencia del bus PCIe.
3. **Muestreo por Mini-batches:** Batches de 512 a 1024 muestras para saturar los Compute Units de la RTX 3060.

---

## 6. Fases del Proyecto y Entregables

```
Fase 1: Motor del Juego  ──>  Fase 2: Red y Pipeline DRL  ──>  Fase 3: Optimización GPU  ──>  Fase 4: Análisis
(Gymnasium / NumPy)           (DQN / Replay Buffer)           (Ampere / CUDA / Batching)       (Métricas / Visualizador)
```

* **Hito 1 (Motor y Validación):** Implementación de la clase `Game2048Env` con lógica vectorizada. Pruebas unitarias de fusiones, desplazamientos y detección de fin de partida.
* **Hito 2 (Línea Base DRL):** Implementación de DQN canónico con red densa / convolucional básica en PyTorch. Validación del bucle `zero_grad() -> forward -> backward() -> step()`.
* **Hito 3 (Aceleración y Dueling DQN):** Integración de arquitectura Dueling, Double DQN (DDQN) y ejecución del entrenamiento en GPU local.
* **Hito 4 (Visualización y Resultados):** Exportación de curvas de pérdida (*Loss*), evolución de la puntuación máxima alcanzada por episodio y script de reproducción interactiva para evaluar al agente entrenado.

---

## 7. Métricas de Éxito

1. **Tasa de Consecución de Fichas:**
   * Ficha 512: Lograda en el $> 90\%$ de las partidas tras convergencia.
   * Ficha 1024: Lograda en el $> 70\%$ de las partidas.
   * Ficha 2048: Lograda en al menos un $40\%$ de las partidas sin heurísticas externas.
2. **Eficiencia Computacional:**
   * Tasa de procesamiento durante entrenamiento superior a **1.500 pasos por segundo** con actualización periódica de pesos en la RTX 3060.