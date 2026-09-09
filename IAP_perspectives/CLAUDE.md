# Proyecto: Rainbow Hologram (dot-matrix) para EBL

## Objetivo
Generar un layout `.gds` para producción en Electron Beam Lithography (EBL): rejillas de
difracción con ciertos periodos y duty cycles que, al observarse desde distintos ángulos
(-10°, -5°, 0°, 5°, 10°), generan la sensación de movimiento de un objeto 3D (efecto
"rainbow hologram" / dot-matrix).

Referencia metodológica: `rainbow-ebeam.pdf`. Diferencia clave respecto al paper: **no
usamos curvatura para corregir el blur** (eso queda para una etapa avanzada posterior).

## Etapas del proyecto

1. **Perspectivas 3D** — ✅ completado (Blender, fuera de este repo).
   Imágenes: `view_-10deg.png`, `view_-05deg.png`, `view_000deg.png`,
   `view_+05deg.png`, `view_+10deg.png`.

2. **Tabla de parámetros por macro-píxel** (`rainbow_grattings_generator.py`) — ✅ implementado.
   Recorre cada imagen/canal y calcula, por macro-píxel, el periodo y el duty cycle de la
   mini-rejilla correspondiente. Escribe una tabla larga en `rainbow_hologram_layer_table.csv`.
   **No dibuja geometría ni toca GDS.**

3. **Generador de GDS a partir del CSV** — ✅ completado .
   Script separado que lea la tabla del paso 2 y dibuje las rejillas reales,
   exportando a `.gds` con `gdsfactory` (verificación con `klayout`).

4. **(Avanzado, fuera de alcance por ahora)** Corrección de curvatura para igualar el
   blur angular entre canales R/G/B (rejillas en arco, como en el paper de referencia).

5. **(Avanzado, fuera de alcance por ahora)** Corrección de dosis por efecto de
   proximidad en el e-beam.

6. **(Avanzado, fuera de alcance por ahora)** Optimización de tamaño de archivo del GDS
   (estilo GDSII más liviano / GDoeSII). (Después de las pruebas, el archivo no tiene un peso considerable)

## Parámetros experimentales

- Celda madre: 60 µm × 60 µm.
- Dentro de cada celda: 3 columnas (canales R, G, B) × 5 filas (una por perspectiva/ángulo).
- Cada mini-rejilla individual: 5 µm × 5 µm (`ax`, `ay`).
- Ángulos de observación: `alpha_i = -10°, -5°, 0°, 5°, 10°`.
- Ángulo de iluminación: `beta = 45°` (constante para todo el experimento).
- Longitudes de onda de referencia (nm): R = 620.0, G = 540.0, B = 470.0.

## Ecuaciones

**Periodo de rejilla** (fijo por combinación imagen+canal):
```
d_i,c = lambda_c / (sin(alpha_i) + sin(beta))
```

**Duty cycle** (varía por píxel, a partir del brillo normalizado 0..1 del canal de color):
```
h/d = arccos(1 - 2*(I/I0)) / (2*pi)
```

Optimización: si un píxel es negro, se puede saltar directamente (zona vacía).

## Herramientas
- `gdsfactory` — generación de la geometría GDS.
- `klayout` — visualización/verificación del `.gds` resultante.

## Notas y decisiones pendientes de verificar
- **Convención de signos** en la ecuación del periodo: se asume
  `d*(sin(alpha)+sin(beta)) = lambda` con alpha y beta del mismo lado de la normal.
  Si la geometría real tiene alpha y beta en lados opuestos, cambiar a
  `d*(sin(alpha)-sin(beta)) = lambda`. Verificar antes de fabricar.
- Mantener la geometría simple en esta etapa; la optimización de tamaño es la etapa 6.

## Estado actual
- Etapa 2 (`rainbow_grattings_generator.py`) funcional.
- Diseñar e implementar el script de la etapa 3 (CSV → GDS) funcional.
