## SAMUEL MARTINEZ NC 0096
import cv2
import numpy as np
import os
import csv

# 1. Configuración inicial 
ruta_script = os.path.dirname(os.path.abspath(__file__))
ruta_raiz = os.path.dirname(ruta_script)

carpeta_original = os.path.join(ruta_raiz, "img-original-0096")
carpeta_preparadas = os.path.join(ruta_raiz, "img-preparadas-0096")
carpeta_resultados = os.path.join(ruta_raiz, "resultados-img-0096")

ANCHO_DESTINO = 224
ALTO_DESTINO = 224

# Crear carpetas de salida si no existen
os.makedirs(carpeta_preparadas, exist_ok=True)
os.makedirs(carpeta_resultados, exist_ok=True)

categorias = ["guitarras", "violines"]
reporte = []
imagenes_procesadas = 0
imagenes_aprobadas = 0

print("🚀 Iniciando preparación de imágenes...\n")

# 2. Recorrer cada categoría y sus archivos
for categoria in categorias:
    ruta_origen = os.path.join(carpeta_original, categoria)
    ruta_destino = os.path.join(carpeta_preparadas, categoria)
    
    os.makedirs(ruta_destino, exist_ok=True)
    
    if not os.path.exists(ruta_origen):
        print(f"⚠️ La carpeta {ruta_origen} no existe. Saltando...")
        continue
        
    for nombre_archivo in os.listdir(ruta_origen):
        # Filtrar solo archivos de imagen válidos
        if not nombre_archivo.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            continue
            
        ruta_imagen = os.path.join(ruta_origen, nombre_archivo)
        nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
        
        print(f"📸 Procesando: {nombre_archivo}")
        
        # 3. Leer la imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            print(f"   ❌ Error: No se pudo leer la imagen.")
            reporte.append([categoria, nombre_archivo, "ERROR", "-", "-", "-", "-", "-", "-", "REVISAR"])
            continue
            
        imagenes_procesadas += 1
        alto_orig, ancho_orig, canales = imagen.shape
        print(f"   📏 Dimensiones originales: {ancho_orig} x {alto_orig}")
        
        # 4. Redimensionar la imagen
        imagen_redimensionada = cv2.resize(imagen, (ANCHO_DESTINO, ALTO_DESTINO), interpolation=cv2.INTER_AREA)
        print(f"   📐 Dimensiones nuevas: {ANCHO_DESTINO} x {ALTO_DESTINO}")
        
        # Guardar la imagen redimensionada como JPG
        ruta_jpg = os.path.join(ruta_destino, f"{nombre_sin_ext}.jpg")
        cv2.imwrite(ruta_jpg, imagen_redimensionada)
        
        # 5. Normalizar los valores de los píxeles (de 0-255 a 0.0-1.0)
        imagen_normalizada = imagen_redimensionada.astype(np.float32) / 255.0
        val_min = float(np.min(imagen_normalizada))
        val_max = float(np.max(imagen_normalizada))
        print(f"   🔢 Rango normalizado: {val_min:.3f} a {val_max:.3f}")
        
        # Guardar los datos numéricos normalizados como archivo .npy
        ruta_npy = os.path.join(ruta_destino, f"{nombre_sin_ext}.npy")
        np.save(ruta_npy, imagen_normalizada)
        
        # 6. Evaluar automáticamente el resultado
        dimensiones_correctas = (imagen_redimensionada.shape[0] == ALTO_DESTINO and 
                                 imagen_redimensionada.shape[1] == ANCHO_DESTINO)
        rango_correcto = (val_min >= 0.0 and val_max <= 1.0)
        archivos_guardados = os.path.isfile(ruta_jpg) and os.path.isfile(ruta_npy)
        
        if dimensiones_correctas and rango_correcto and archivos_guardados:
            estado = "APROBADA"
            imagenes_aprobadas += 1
            print(f"   ✅ Evaluación: APROBADA\n")
        else:
            estado = "REVISAR"
            print(f"   ⚠️ Evaluación: REVISAR\n")
            
        # 7. Guardar datos en la lista para el reporte final
        reporte.append([
            categoria, nombre_archivo, "OK", ancho_orig, alto_orig,
            ANCHO_DESTINO, ALTO_DESTINO, round(val_min, 3), round(val_max, 3), estado
        ])

# 8. Generar el reporte en formato CSV
ruta_reporte = os.path.join(carpeta_resultados, "reporte_preparacion.csv")
with open(ruta_reporte, "w", newline="", encoding="utf-8") as archivo:
    escritor = csv.writer(archivo)
    escritor.writerow([
        "Categoría", "Archivo original", "Lectura", 
        "Ancho original", "Alto original", "Ancho preparado", "Alto preparado", 
        "Valor mínimo norm.", "Valor máximo norm.", "Evaluación"
    ])
    escritor.writerows(reporte)

# 9. Resumen final en consola
print("=" * 45)
print("🎉 RESUMEN DE LA PRÁCTICA")
print("=" * 45)
print(f"Imágenes procesadas: {imagenes_procesadas}")
print(f"Imágenes aprobadas:  {imagenes_aprobadas}")
print(f"Imágenes a revisar:  {imagenes_procesadas - imagenes_aprobadas}")
print(f"\n📁 Reporte guardado en: {ruta_reporte}")
print("✅ PRÁCTICA TERMINADA EXITOSAMENTE SAMUEL MARTINEZ NC 0096")