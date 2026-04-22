# Dataset: Detección de Fraude en Seguros de Automóvil — AutoGuard Seguros

---

## claims.csv

Tabla principal. Una fila por siniestro reportado a la aseguradora.

- `claim_id` — Identificador único del siniestro. Clave primaria. Formato: `CLM_XXXXX`.
- `policy_id` — Póliza bajo la que se presenta el siniestro. FK → `policies.policy_id`.
- `customer_id` — Asegurado que presenta la reclamación. FK → `customers.customer_id`.
- `vehicle_id` — Vehículo involucrado en el siniestro. FK → `vehicles.vehicle_id`.
- `agent_id` — Mediador/representante que gestiona el siniestro. Formato: `AGT_XXXXX`. No existe tabla maestra de agentes; este campo solo permite agrupar siniestros por mediador.
- `accident_datetime` — Fecha y hora del accidente. Formato: `YYYY-MM-DD HH:MM:SS`.
- `claim_datetime` — Fecha y hora en que se registra la reclamación. Siempre posterior o igual a `accident_datetime`. Formato: `YYYY-MM-DD HH:MM:SS`.
- `fault` — Parte responsable del accidente según la declaración. Valores: `Policy Holder`, `Third Party`.
- `accident_area` — Tipo de zona donde ocurrió el accidente. Valores: `Urban`, `Rural`, `Suburban`, `Highway`, `Parking`.
- `accident_description` — Descripción libre del tipo de siniestro. Ejemplos: `rear-end collision`, `hit and run`, `parking damage`, `theft attempt`, etc. Aproximadamente 20 valores distintos.
- `accident_latitude` — Latitud de la ubicación del accidente (coordenadas en España).
- `accident_longitude` — Longitud de la ubicación del accidente.
- `police_report_filed` — Si se presentó atestado policial. Valores: `Yes`, `No`, `Unknown`.
- `witness_present` — Si hubo testigos del accidente. Valores: `Yes`, `No`, `Unknown`.
- `number_of_supplements` — Número de documentos suplementarios aportados por el reclamante (0–7).
- `claimed_amount_eur` — Importe reclamado en euros. Distribución log-normal; media ~2.900 €, rango desde ~100 € hasta >50.000 €.
- `repair_workshop` — Nombre del taller de reparación. Alta cardinalidad (~400 talleres distintos). Formato libre: `Taller García - Madrid`, `Garage López - Barcelona`, etc.

---

## customers.csv

Una fila por asegurado.

- `customer_id` — Identificador único del cliente. Clave primaria. Formato: `CUS_XXXXX`.
- `full_name` — Nombre completo del asegurado. Formato: `Nombre Apellido1 Apellido2`. Alta cardinalidad.
- `sex` — Sexo. Valores: `Male`, `Female`.
- `marital_status` — Estado civil. Valores: `Single`, `Married`, `Divorced`, `Widow`.
- `date_of_birth` — Fecha de nacimiento. Formato: `YYYY-MM-DD`. Rango: 1940–2005.
- `email` — Dirección de correo electrónico. Alta cardinalidad.
- `phone` — Número de teléfono. Formato: `+34 6XX XXX XXX`. Alta cardinalidad.
- `address` — Dirección postal. Formato: `NNN Calle Tipo`. Alta cardinalidad.
- `city` — Ciudad de residencia. ~40 ciudades españolas.
- `province` — Provincia. ~30 provincias.
- `postal_code` — Código postal (5 dígitos, puede perder el cero inicial al leerse como entero).
- `last_address_change_date` — Fecha del último cambio de dirección. Formato: `YYYY-MM-DD`. Vacío si el cliente nunca cambió de dirección.

---

## policies.csv

Una fila por póliza de seguro. Un cliente puede tener varias pólizas.

- `policy_id` — Identificador único de la póliza. Clave primaria. Formato: `POL_XXXXX`.
- `customer_id` — Cliente titular de la póliza. FK → `customers.customer_id`.
- `policy_type` — Tipo completo de la póliza. Formato: `Categoría - Cobertura`. Ejemplos: `Sport - Liability`, `Sedan - Collision`, `Utility - All Perils`. 9 valores distintos.
- `base_policy` — Tipo de cobertura base. Valores: `Liability`, `Collision`, `All Perils`. Es la parte derecha de `policy_type`.
- `deductible` — Franquicia de la póliza en euros. Valores: 300, 400, 500, 700.
- `annual_premium_eur` — Prima anual en euros. Distribución log-normal; media ~650 €.
- `policy_start_date` — Fecha de inicio de la póliza. Formato: `YYYY-MM-DD`.
- `policy_end_date` — Fecha de fin de la póliza. Siempre 365 días después de `policy_start_date`. Formato: `YYYY-MM-DD`.
- `past_number_of_claims` — Número de siniestros previos declarados al contratar la póliza. Atención: es un string categórico, no un entero. Valores: `none`, `1`, `2`, `3`, `4`, `more than 4`.
- `number_of_cars` — Número de vehículos cubiertos por la póliza (1–8).

---

## vehicles.csv

Una fila por vehículo asegurado. Un vehículo está vinculado a una póliza.

- `vehicle_id` — Identificador único del vehículo. Clave primaria. Formato: `VEH_XXXXX`.
- `policy_id` — Póliza a la que pertenece el vehículo. FK → `policies.policy_id`.
- `license_plate` — Matrícula del vehículo. Formato: `NNNN LLL`. Alta cardinalidad.
- `make` — Marca del vehículo. ~30 marcas: Toyota, BMW, Peugeot, SEAT, Dacia, etc.
- `model` — Modelo del vehículo. ~150 modelos distintos: Corolla, Golf, 308, León, etc. Alta cardinalidad.
- `manufacture_year` — Año de fabricación del vehículo (2005–2024).
- `vehicle_category` — Categoría del vehículo. Valores: `Sport`, `Sedan`, `Utility`, `SUV`, `Van`, `Coupe`.
- `purchase_price_eur` — Precio de compra del vehículo en euros. Distribución log-normal; media ~18.000 €.
- `color` — Color del vehículo. Valores: `Blanco`, `Negro`, `Gris`, `Plata`, `Rojo`, `Azul`, `Verde`, `Beige`, `Marrón`, `Naranja`, `Amarillo`.
- `odometer_km` — Lectura del odómetro en kilómetros en el momento del registro (500–280.000).
- `driver_rating` — Calificación del conductor principal (1–4, donde 1 = mejor historial).

---

## claim_reviews.csv

Historial de auditorías y revisiones. Cada siniestro tiene entre 1 y 3 filas, una por ronda de revisión. Esta tabla se usa exclusivamente para construir el target; sus columnas no deben usarse como features del modelo.

- `review_id` — Identificador único de la revisión. Formato: `REV_XXXXX_N` donde N es el número de ronda.
- `claim_id` — Siniestro revisado. FK → `claims.claim_id`.
- `review_round` — Ronda de revisión. Valores: 1, 2, 3.
- `review_datetime` — Fecha y hora de la revisión. Formato: `YYYY-MM-DD HH:MM:SS`. Siempre posterior a `claim_datetime`.
- `reviewer_id` — Identificador del revisor. Formato: `R-NNNN` (rondas 1-2) o `SIU-NNN` (ronda 3).
- `review_type` — Tipo de revisión. Valores: `Initial Triage` (ronda 1), `Expert Assessment` (ronda 2), `SIU Investigation` (ronda 3).
- `auto_risk_score` — Puntuación de riesgo automática (0–100). Solo presente en ronda 1. Vacío en rondas 2 y 3.
- `triage_result` — Resultado del triaje. Solo en ronda 1. Valores: `Green - Proceed`, `Yellow - Standard Review`, `Orange - Escalated Review`, `Red - Full Investigation`. Vacío en rondas 2 y 3.
- `reviewer_notes` — Notas textuales del revisor. Texto libre con observaciones sobre el caso.
- `perito_id` — Identificador del perito asignado. Formato: `P-NNNN`. Solo presente en rondas 2 y 3. Vacío en ronda 1.
- `perito_physical_inspection` — Si el perito realizó inspección física del vehículo. Valores: `Yes`, `No`. Solo en rondas 2 y 3. Vacío en ronda 1.
- `damage_consistency_rating` — Valoración del perito sobre la coherencia entre los daños reportados y los observados. Escala 1.0–5.0 (5 = totalmente consistente). Solo en ronda 2. Vacío en rondas 1 y 3.
- `documentation_completeness_pct` — Porcentaje de documentación completa según el perito (0–100). Solo en ronda 2. Vacío en rondas 1 y 3.
- `recommended_action` — Acción recomendada al cierre de la ronda. Los valores posibles varían según la ronda. En ronda 1: `Proceed`, `Standard Review`, `Escalated Review`, `Full Investigation`. En ronda 2: `Approve`, `Approve with Conditions`, `Request Additional Documentation`, `Refer to Special Investigations Unit`, `Recommend Denial`. En ronda 3: `Claim Approved`, `Claim Settled at Reduced Amount`, `Claim Denied`, `Pending Review`, `Pending Litigation`, `Claim Withdrawn by Claimant`.
- `days_since_claim` — Días transcurridos desde `claim_datetime` hasta la fecha de esta revisión.

Cobertura por ronda: ronda 1 cubre el 100% de los siniestros (campos `auto_risk_score`, `triage_result`); ronda 2 cubre ~57% (campos `perito_id`, `perito_physical_inspection`, `damage_consistency_rating`, `documentation_completeness_pct`); ronda 3 cubre ~13% (campos `perito_id`, `perito_physical_inspection`).
