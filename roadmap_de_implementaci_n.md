# Roadmap de Implementación: Agente Autónomo para 2048 con Deep Reinforcement Learning y GPU

Este documento describe la hoja de ruta técnica detallada, organizada en fases consecutivas, paquetes de trabajo (Work Packages) y tareas atómicas para la construcción, entrenamiento, validación y despliegue del agente de Aprendizaje por Refuerzo Profundo (DRL) para el entorno 2048.

---

## Diagrama General de Fases y Dependencias

```
+---------------------------------------------------------------------------------------+
| FASE 1: Motor del Entorno y Representación Vectorial                                  |
| (Lógica pura NumPy / Gymnasium, One-Hot Encoding, Action Masking)                      |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| FASE 2: Arquitectura Neuronal y Buffer de Repetición                                  |
| (Dueling DQN, Conv2d / ResNet blocks, Prioritized Replay Buffer, TF32 setup)          |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| FASE 3: Bucle de Entrenamiento DRL y Estabilización Numérica                          |
| (Double DQN Loss, Huber Loss, Epsilon-Greedy adaptativo, Target Network sync)         |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| FASE 4: Optimización de Rendimiento y Pipeline de Cómputo                             |
| (Vectorización de Entornos paralelos en GPU, Pinned Memory, Profiling PyTorch/CUDA)   |
+-------------------------------------------+-------------------------------------------+
                                            |
                                            v
+---------------------------------------------------------------------------------------+
| FASE 5: Evaluación, Telemetría y Renderizado de Inferencia                            |
| (TensorBoard metrics, Checkpointing de pesos, Visualizador CLI / Web headless)        |
+---------------------------------------------------------------------------------------+
```

---

## Fase 1: Motor del Entorno Headless y Representación Vectorial

El objetivo de esta fase es disponer de un simulador estricto, determinista en sus reglas pero estocástico en la aparición de fichas, con latencia mínima por paso y sin dependencias gráficas.

### WP 1.1: Mecánicas del Núcleo del Juego (`Game2048Core`)
- **Tarea 1.1.1: Representación de Estado Interno**
  - Matriz bidimensional de tipo `np.int32` o `np.int64` de dimensiones fijas $(4, 4)$.
  - Inicialización: matriz de ceros con colocación aleatoria de dos fichas: valor 2 con probabilidad $p = 0.9$, valor 4 con probabilidad $p = 0.1$.
- **Tarea 1.1.2: Primitiva de Desplazamiento y Fusión Unidireccional (Compresión Izquierda)**
  - Implementar una función pura optimizada que opere sobre una fila de longitud 4:
    1. Filtrar ceros hacia la derecha (compactación).
    2. Colapsar pares adyacentes idénticos de izquierda a derecha. Regla crítica de 2048: una casilla fusionada en un turno no puede volver a fusionarse en el mismo turno (ejemplo: `[2, 2, 2, 2]` se convierte en `[4, 4, 0, 0]`, no en `[8, 0, 0, 0]`).
    3. Acumular los puntos exactos generados por las fusiones para el cálculo del retorno.
    4. Rellenar con ceros hasta restaurar longitud 4.
- **Tarea 1.1.3: Generalización Matricial por Rotación y Reflexión**
  - Reutilizar la primitiva de compresión izquierda para las 4 direcciones mediante transformaciones ortogonales de NumPy (cero duplicación de código de colisión):
    - **Izquierda ($\mathcal{A}=0$):** Operación directa sobre las filas.
    - **Derecha ($\mathcal{A}=1$):** Inversión horizontal con `np.fliplr()`, compresión y reversión.
    - **Arriba ($\mathcal{A}=2$):** Transposición con `np.transpose()`, compresión y transposición inversa.
    - **Abajo ($\mathcal{A}=3$):** Rotación o transposición combinada con `np.flipud()`.
- **Tarea 1.1.4: Detección de Modificación de Tablero**
  - Comprobación booleana `not np.array_equal(tablero_anterior, tablero_nuevo)`.
  - Si no hay cambio, el movimiento es catalogado como **inválido/ilegal**.
- **Tarea 1.1.5: Mecánica de Aparición de Nuevas Fichas**
  - Detección de índices de celdas vacías (`tablero == 0`).
  - Selección uniforme aleatoria de una celda vacía y asignación ponderada ($2$ con $90\%$, $4$ con $10\%$).
- **Tarea 1.1.6: Condición de Fin de Partida (`Terminal Check`)**
  - La partida termina si y solo si:
    1. No existen celdas vacías (`np.count_nonzero(tablero == 0) == 0`).
    2. Ningún par de celdas adyacentes horizontal o verticalmente comparte el mismo valor (ausencia total de fusiones legales).

### WP 1.2: Envoltorio Gymnasium y Espacios de Entrada (`Game2048Env`)
- **Tarea 1.2.1: Definición de Interfaces Estándar**
  - Herencia de `gymnasium.Env`.
  - Definición de `action_space = spaces.Discrete(4)`.
  - Definición de `observation_space`: espacio de tensores multidimensionales continuos o binarios.
- **Tarea 1.2.2: Máscara de Acciones Legales (`Action Masking`)**
  - Implementar método `get_action_mask(state) -> np.ndarray[bool, 4]` que simule en memoria las 4 direcciones y devuelva un vector booleano indicando qué acciones alteran el tablero sin comprometer el estado actual.
- **Tarea 1.2.3: Codificación de Observaciones (One-Hot Multi-Canal)**
  - Transformar la matriz $(4, 4)$ en un tensor de dimensiones $(16, 4, 4)$ o $(17, 4, 4)$:
    - Canal 0: Representa celdas vacías ($2^0 \implies 0$).
    - Canales $1 \dots 16$: Representan potencias $2^1, 2^2, \dots, 2^{16}$ (desde 2 hasta 65.536).
  - Cada posición $(c, i, j)$ contendrá un valor binario $1.0$ si la casilla $(i, j)$ almacena el valor correspondiente al canal $c$, y $0.0$ en caso contrario.
- **Tarea 1.2.4: Función de Recompensa Escalar y Logarítmica**
  - Cálculo de la recompensa por paso $r_t$:
    $$r_t = \begin{cases} 
    \sum \log_2(\text{valor\_fusión}) & \text{si hubo fusiones legales} \\
    0.0 & \text{si el movimiento fue legal pero sin fusiones} \\
    -5.0 \text{ a } -10.0 & \text{si la acción fue ilegal (si no se usa action masking)}
    \end{cases}$$
- **Tarea 1.2.5: Pruebas Unitarias del Motor**
  - Crear suite de tests con `pytest` validando:
    - Casos extremos de fusiones en fila (ejemplo: `[4, 4, 4, 4]` $\to$ `[8, 8, 0, 0]`).
    - Detección precisa de bloqueos y tableros llenos con movimientos remanentes.
    - Rendimiento sintético de simulación ($> 15.000\text{ steps/s}$ en un solo hilo de CPU).

---

## Fase 2: Arquitectura de Red Neuronal y Memoria de Experiencia

El propósito de esta fase es diseñar el extractor de características topológicas y el sistema de almacenamiento desacoplado de transiciones.

### WP 2.1: Red Neuronal Dueling Convolucional (`Dueling2048Net`)
- **Tarea 2.1.1: Capa de Entrada y Extracción Espacial**
  - Entrada: Tensor $\mathbf{X} \in \mathbb{R}^{B \times 16 \times 4 \times 4}$.
  - Dado el tamaño reducido del tablero ($4 \times 4$), los filtros tradicionales grandes ($5 \times 5$ o $7 \times 7$) provocan colapso prematuro de dimensiones. Se implementarán dos ramas paralelas o convoluciones asimétricas:
    - **Rama Horizontal:** `Conv2d(16, 128, kernel_size=(1, 2))` para extraer dependencias adyacentes en filas.
    - **Rama Vertical:** `Conv2d(16, 128, kernel_size=(2, 1))` para extraer dependencias adyacentes en columnas.
    - **Rama Cuadrada:** `Conv2d(16, 128, kernel_size=(2, 2))` para captar bloques compactos locales.
- **Tarea 2.1.2: Capas de Fusión y Normalización**
  - Concatenación de características extraídas por las ramas convolucionales.
  - Activaciones no lineales: `LeakyReLU(negative_slope=0.01)` para evitar el problema de neuronas muertas.
  - Proyección mediante capas lineales intermedias de 512 neuronas con regularización leve o Dropout bajo ($p=0.05$).
- **Tarea 2.1.3: Separación Dueling (Value & Advantage Streams)**
  - **Sub-red de Valor $V(s)$:** Capa densa de $512 \to 128 \to 1$ escalar que estima la calidad intrínseca del tablero independientemente de la acción.
  - **Sub-red de Ventaja $A(s, a)$:** Capa densa de $512 \to 128 \to 4$ que estima la ventaja relativa de cada una de las 4 direcciones.
- **Tarea 2.1.4: Agregación Centrada de Q-Values**
  - Implementación en el método `forward(x, action_mask=None)` de la ecuación:
    $$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$$
  - Si se proporciona `action_mask`, aplicar un enmascaramiento aditivo a los logits de salida antes de seleccionar la acción:
    $$Q_{\text{masked}}(s, a) = \begin{cases} Q(s, a) & \text{si } \text{mask}(a) = \text{True} \\ -\infty \text{ (o } -10^9\text{)} & \text{si } \text{mask}(a) = \text{False} \end{cases}$$

### WP 2.2: Buffer de Repetición de Experiencias (`PrioritizedReplayBuffer`)
- **Tarea 2.2.1: Estructura de Almacenamiento Circular**
  - Capacidad predefinida ($N = 100.000$ a $200.000$ transiciones).
  - Almacenar tuplas compactas: `(state, action, reward, next_state, done, action_mask)`.
  - Guardar estados en memoria RAM como matrices empaquetadas en `uint8` (índices de potencias de 2) para reducir la huella de memoria por debajo de 1.5 GB.
- **Tarea 2.2.2: Muestreo Priorizado (PER - Prioritized Experience Replay)**
  - Implementación basada en árbol de suma binario (*SumTree*) con complejidad temporal $\mathcal{O}(\log N)$.
  - Prioridad de cada transición calculada a partir de su error temporal TD:
    $$P(i) = \frac{(|\delta_i| + \epsilon)^\alpha}{\sum_k (|\delta_k| + \epsilon)^\alpha}$$
    donde $\alpha \in [0.5, 0.7]$ y $\epsilon = 10^{-5}$.
- **Tarea 2.2.3: Corrección por Muestreo con Pesos de Importancia (IS Weights)**
  - Cálculo de pesos $w_i = (N \cdot P(i))^{-\beta} / \max(w)$ para compensar el sesgo de muestreo no uniforme, con $\beta$ creciendo linealmente desde $0.4$ hasta $1.0$ a lo largo del entrenamiento.

---

## Fase 3: Bucle de Entrenamiento DRL y Estabilización

El objetivo de esta fase es implementar el algoritmo Double DQN acoplado a la optimización por gradiente en GPU.

### WP 3.1: Algoritmo de Aprendizaje Double DQN (DDQN)
- **Tarea 3.1.1: Doble Red Neuronal (Policy Net & Target Net)**
  - `policy_net`: Red en entrenamiento constante, cuyos pesos se actualizan en cada paso de optimización.
  - `target_net`: Copia estricta de la arquitectura utilizada para estimar los valores futuros de Bellman sin inestabilidades de retroalimentación.
- **Tarea 3.1.2: Cálculo del Objetivo de Bellman DDQN**
  - Selección de la mejor acción futura mediante la red de política:
    $$a^* = \arg\max_{a'} Q(s_{t+1}, a'; \theta_{\text{policy}})$$
  - Evaluación del valor de dicha acción mediante la red objetivo:
    $$y_t = r_t + (1 - \text{done}) \cdot \gamma \cdot Q(s_{t+1}, a^*; \theta_{\text{target}})$$
    con factor de descuento $\gamma = 0.99$.
- **Tarea 3.1.3: Función de Pérdida Robusta**
  - Cálculo de la pérdida temporal ponderada mediante **Smooth L1 Loss (Huber Loss)**:
    $$\mathcal{L}(\theta) = \frac{1}{B} \sum_{i=1}^B w_i \cdot \text{Huber}(y_i - Q(s_i, a_i; \theta_{\text{policy}}))$$
  - Actualización de prioridades del *SumTree* con los nuevos errores absolutos $|\delta_i| = |y_i - Q(s_i, a_i)|$.

### WP 3.2: Estrategia de Exploración y Dinámica de Optimización
- **Tarea 3.2.1: Política Epsilon-Greedy Decaída con Selección Legal**
  - Parámetros: $\epsilon_{\text{start}} = 1.0$, $\epsilon_{\text{min}} = 0.01$, decaimiento exponencial o coseno a lo largo de $150.000$ pasos.
  - En fase de exploración ($\text{rand}() < \epsilon$), seleccionar aleatoriamente **únicamente** entre las acciones legales devueltas por la máscara booleana.
- **Tarea 3.2.2: Sincronización de Pesos Target (Polyak Averaging vs. Hard Update)**
  - Alternativa A (Hard Update): Copiar parámetros cada $C = 1.000$ pasos de gradiente (`target_net.load_state_dict(policy_net.state_dict())`).
  - Alternativa B (Soft Update / Polyak): Actualización continua tras cada paso:
    $$\theta_{\text{target}} \leftarrow \tau \theta_{\text{policy}} + (1 - \tau) \theta_{\text{target}}, \quad \text{con } \tau = 0.005$$
- **Tarea 3.2.3: Configuración del Optimizador**
  - Optimizador **AdamW** con tasa de aprendizaje $\eta = 10^{-4}$ y decaimiento de peso $10^{-4}$.
  - **Gradient Clipping:** Recorte de la norma máxima del gradiente (`torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)`) para prevenir explosión numérica.

---

## Fase 4: Optimización de Rendimiento y Pipeline de Cómputo GPU

El propósito de esta fase es explotar la arquitectura Ampere (RTX 3060) para maximizar los pasos por segundo del sistema completo.

### WP 4.1: Aceleración Ampere y Pipeline de Memoria
- **Tarea 4.1.1: Activación de Precisión TensorFloat-32 (TF32)**
  - Configurar las directivas globales de backend de CUDA:
    ```python
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    ```
- **Tarea 4.1.2: Transferencias Asíncronas Host-to-Device (PCIe)**
  - Asignación de batches con memoria fijada en RAM (`pin_memory=True`).
  - Transferencia no bloqueante hacia la GPU mediante `tensor.to(device, non_blocking=True)`.
- **Tarea 4.1.3: Vectorización Multi-Entorno (`AsyncVectorEnv`)**
  - Instanciar entre 8 y 32 entornos paralelos de 2048 en procesos independientes de CPU o en un lote unificado.
  - Recolección síncrona de transiciones: mientras la GPU computa una pasada hacia atrás del lote de optimización, los entornos de CPU avanzan el siguiente paso de simulación.

### WP 4.2: Auditoría y Profiling de Carga
- **Tarea 4.2.1: Medición de Balance CPU/GPU**
  - Uso de `torch.profiler` para medir el tiempo consumido en:
    1. Simulación de pasos de juego (CPU).
    2. Conversión One-Hot y transferencias por bus PCIe.
    3. Pasada hacia adelante y cálculo de gradientes en Tensor Cores (GPU).
- **Tarea 4.2.2: Ajuste del Ratio de Actualización (*Replay Ratio*)**
  - Calibrar la relación entre pasos de entorno y pasos de gradiente (típicamente 4 pasos de simulación por cada 1 actualización de pesos) para evitar sobreajuste y desaturación del pipeline.

---

## Fase 5: Telemetría, Evaluación y Exportación

El objetivo de esta fase es cuantificar rigurosamente la convergencia del agente y habilitar mecanismos de inspección.

### WP 5.1: Registro de Métricas y Telemetría
- **Tarea 5.1.1: Integración con TensorBoard / Weights & Biases**
  - Registro por episodio:
    - Puntuación final alcanzada ($Score$).
    - Ficha máxima obtenida ($Tile_{\max} \in [128, 256, 512, 1024, 2048, 4096]$).
    - Longitud de partida (número de pasos de supervivencia).
    - Pérdida media del optimizador ($\mathcal{L}_{\text{TD}}$).
    - Valor promedio estimado de $Q(s, a)$.
    - Tasa de decaimiento del parámetro $\epsilon$.
- **Tarea 5.1.2: Distribución Histográfica de Fichas Mayores**
  - Cálculo de la tasa acumulada de éxito en ventanas móviles de 1.000 partidas (% de partidas que alcanzan $\ge 1024$ y $\ge 2048$).

### WP 5.2: Checkpointing y Recuperación
- **Tarea 5.2.1: Guardado Atómico de Pesos**
  - Guardar checkpoints periódicos (`checkpoint_epoch_X.pt`) conteniendo:
    - `policy_net.state_dict()`
    - `target_net.state_dict()`
    - `optimizer.state_dict()`
    - Historial del replay buffer y valor actual de $\epsilon$.
- **Tarea 5.2.2: Selección de Mejor Modelo**
  - Guardado independiente del modelo con mayor puntuación media sobre 100 partidas de evaluación greedy ($\epsilon = 0$).

### WP 5.3: Reproducción Visual e Inferencia Headless
- **Tarea 5.3.1: Reproductor por Consola (CLI Viewer)**
  - Script liviano que cargue los pesos del mejor modelo, configure el entorno en modo greedy puro y dibuje el tablero en terminal en tiempo real con colores ANSI por cada potencia de 2.
- **Tarea 5.3.2: Exportación a Formato Abierto (ONNX)**
  - Conversión del modelo a `model_2048.onnx` para permitir su ejecución en cualquier navegador web o aplicación externa sin dependencias de PyTorch.

---

## Cronograma de Ejecución y Dependencias Temporales

| Fase | Semana / Módulo | Hitos Principales | Dependencias Previas |
| :--- | :---: | :--- | :--- |
| **Fase 1** | Semana 1 | Motor 2048 funcional, pruebas unitarias y generación de tensores One-Hot. | Ninguna |
| **Fase 2** | Semana 2 | Red Dueling DQN implementada en PyTorch y buffer circular validado. | Fase 1 completada |
| **Fase 3** | Semanas 3-4 | Bucle DDQN cerrado, primera convergencia hacia ficha 512/1024. | Fases 1 y 2 completadas |
| **Fase 4** | Semanas 4-5 | Optimización con TF32, saturación de GPU y paralelización de pasos. | Fase 3 funcional |
| **Fase 5** | Semana 5+ | Métricas finales, consecución consistente de ficha 2048 y exportación ONNX. | Todas las fases anteriores |