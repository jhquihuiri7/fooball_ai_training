# Protocolo de grabación de la campaña (ML-06)

Qué se hace en cada partido para que la grabación sirva de dataset. El manifiesto
(`matches/_plantilla.yaml`, validado por `ftrain/manifest.py`) es la parte que se
rellena; esto es la parte que se ejecuta. **El propietario lo revisó y lo aceptó el
2026-10-09, antes del primer partido.**

## Lista previa (en el montaje, antes del saque)

- [ ] **Hora**: los dos móviles con la hora automática (NTP) y el reloj del soporte
      enlazado (la tarjeta Cámaras en verde con reloj).
- [ ] **«Guardar vídeo» encendido en los dos**: HEVC 4K a 45 Mbit/s con el rigMs
      pintado. La grabación local es el dataset; la emisión es aparte.
- [ ] **Audio** activo (el N0 lleva el ambiente; los eventos se oyen).
- [ ] **Refrigeración activa** puesta en los dos móviles.
- [ ] **Carga por PD** conectada (hub o batería): un partido entero no sale de la
      batería sola.
- [ ] Lentes limpias; soporte nivelado; inclinación apuntada en la hoja de campo.

## La matriz de la campaña

| Dimensión | Objetivo |
|---|---|
| Partidos | 6-10 (los 2 primeros son el piloto, ML-17) |
| Canchas | al menos 3 distintas |
| Luz | día Y focos (y alguna hora mala, `dusk`) |
| Retranqueo | al menos dos: ~6 m y ~10 m |
| Inclinación | al menos dos distintas |
| Cancha sagrada | UNA cancha completa con `split_hint: sagrada`: jamás en train |

## Qué se captura además del partido

1. **La pareja de calibración**: soporte quieto, 10-20 s apuntando a algo con
   detalle, y «SUBIR PARA CALIBRAR» en los dos móviles. El `rig.json` y el
   `band.json` resultantes van al manifiesto.
2. **Las marcas N0**: goles y jugadas desde el mando durante el partido; el registro
   sale con las grabaciones.
3. **3 ráfagas de 10 s de NV12 crudo por cámara y por parte** (IOS-15: lanzar la app
   con `NV12_DUMP_S` puesto). Son la referencia del salto de dominio de ML-18.

## La hoja de campo (una por partido)

```
match_id: ____________________   cancha: ____________   fecha: __________
luz: day / floodlight / dusk     retranqueo: ______ m   altura: ______ m
inclinación izq: ____°  der: ____°      rotación izq: 0/180  der: 0/180
móviles (modelo + iOS): izq ______________  der ______________
calibración subida: sí/no        ráfagas NV12: 1ª ___ 2ª ___ 3ª ___
marcas N0 (goles, min): _____________________________________________
incidencias (cortes, calor, lluvia): _________________________________
```

Al llegar a casa: subir con `ftrain/storage.py` (raw/, nv12/), copiar la plantilla a
`matches/<match_id>.yaml`, rellenarla desde la hoja y commitearla. Un partido sin
manifiesto no existe para la campaña.
