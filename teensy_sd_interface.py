#!/usr/bin/env python3
"""
Interface Python pour communiquer avec le Teensy et récupérer les données de la carte SD.
Reproduit les fonctionnalités du code Arduino sd_card_code.ino
"""

import serial
import serial.tools.list_ports
import time
import os
import sys
from typing import Optional, List, Tuple
import argparse


class TeensySDInterface:
    """Interface pour communiquer avec le Teensy et gérer les fichiers SD."""

    def __init__(self, port: Optional[str] = None, baudrate: int = 115200):
        """
        Initialise l'interface avec le Teensy.

        Args:
            port: Port série à utiliser (auto-détection si None)
            baudrate: Vitesse de communication (défaut: 115200)
        """
        self.port = port
        self.baudrate = baudrate
        self.serial_conn: Optional[serial.Serial] = None

    def find_teensy_port(self) -> Optional[str]:
        """Trouve automatiquement le port du Teensy."""
        ports = serial.tools.list_ports.comports()

        # Cherche les ports qui pourraient être un Teensy
        for port in ports:
            # Teensy apparaît souvent avec ces descriptions
            if any(keyword in port.description.lower() for keyword in
                   ['teensy', 'usb serial', 'usb2.0-serial']):
                return port.device

        # Si pas trouvé, liste tous les ports disponibles
        print("Ports série disponibles:")
        for i, port in enumerate(ports):
            print(f"{i}: {port.device} - {port.description}")

        return None

    def connect(self) -> bool:
        """
        Établit la connexion avec le Teensy.

        Returns:
            True si la connexion réussit, False sinon
        """
        try:
            if self.port is None:
                self.port = self.find_teensy_port()
                if self.port is None:
                    print("Aucun port Teensy trouvé automatiquement.")
                    return False

            print(f"Connexion au port {self.port}...")
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=5.0,
                write_timeout=5.0
            )

            # Attendre que la connexion soit établie
            time.sleep(2)

            # Vider le buffer d'entrée
            self.serial_conn.reset_input_buffer()

            print("Connexion établie avec le Teensy!")
            return True

        except Exception as e:
            print(f"Erreur de connexion: {e}")
            return False

    def disconnect(self):
        """Ferme la connexion série."""
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            print("Connexion fermée.")

    def send_command(self, command: str) -> str:
        """
        Envoie une commande au Teensy et retourne la réponse.

        Args:
            command: Commande à envoyer

        Returns:
            Réponse du Teensy
        """
        if not self.serial_conn or not self.serial_conn.is_open:
            raise Exception("Pas de connexion série active")

        # Envoyer la commande
        self.serial_conn.write(f"{command}\n".encode('utf-8'))
        self.serial_conn.flush()

        # Lire la réponse
        response = ""
        start_time = time.time()

        while time.time() - start_time < 10:  # Timeout de 10 secondes
            if self.serial_conn.in_waiting > 0:
                data = self.serial_conn.read(self.serial_conn.in_waiting)
                response += data.decode('utf-8', errors='ignore')

                # Vérifier si on a reçu une réponse complète
                if response.endswith('\n') or "End of file list" in response:
                    break

            time.sleep(0.01)

        return response.strip()

    def list_files(self) -> List[str]:
        """
        Liste tous les fichiers sur la carte SD.

        Returns:
            Liste des noms de fichiers
        """
        print("Récupération de la liste des fichiers...")
        response = self.send_command("LIST")

        files = []
        lines = response.split('\n')

        for line in lines:
            if line.strip() and not line.startswith("Files on SD card:") and not line.startswith("End of file list"):
                # Extraire le nom du fichier (avant la parenthèse)
                if '(' in line:
                    filename = line.split('(')[0].strip()
                    files.append(filename)

        return files

    def get_file(self, filename: str, output_dir: str = "downloaded_files") -> bool:
        """
        Télécharge un fichier depuis la carte SD.

        Args:
            filename: Nom du fichier à télécharger
            output_dir: Répertoire de destination

        Returns:
            True si le téléchargement réussit, False sinon
        """
        print(f"Téléchargement de {filename}...")

        # Créer le répertoire de destination si nécessaire
        os.makedirs(output_dir, exist_ok=True)

        # Envoyer la commande GET
        response = self.send_command(f"GET {filename}")

        if "ERROR: File not found" in response:
            print(f"Erreur: Fichier {filename} non trouvé")
            return False

        if not response.startswith("SENDING:"):
            print(f"Erreur: Réponse inattendue: {response}")
            return False

        # Extraire la taille du fichier
        try:
            file_size = int(response.split(':')[-1])
            print(f"Taille du fichier: {file_size} bytes")
        except:
            print("Impossible de déterminer la taille du fichier")
            file_size = 0

        # Lire les données du fichier
        file_data = b""
        start_time = time.time()

        while time.time() - start_time < 30:  # Timeout de 30 secondes
            if self.serial_conn.in_waiting > 0:
                data = self.serial_conn.read(self.serial_conn.in_waiting)
                file_data += data

                # Vérifier si on a reçu la fin du fichier
                if b"END_OF_FILE" in file_data:
                    break

            time.sleep(0.01)

        # Nettoyer les données (enlever END_OF_FILE)
        if b"END_OF_FILE" in file_data:
            file_data = file_data.split(b"END_OF_FILE")[0]

        if not file_data:
            print("Aucune donnée reçue")
            return False

        # Sauvegarder le fichier
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'wb') as f:
            f.write(file_data)

        print(f"Fichier sauvegardé: {output_path}")
        print(f"Taille reçue: {len(file_data)} bytes")

        return True

    def get_help(self) -> str:
        """Affiche l'aide des commandes disponibles."""
        return self.send_command("HELP")

    def interactive_mode(self):
        """Mode interactif pour utiliser l'interface."""
        print("\n=== Interface Teensy SD ===")
        print("Commandes disponibles:")
        print("  list - Lister les fichiers")
        print("  get <filename> - Télécharger un fichier")
        print("  help - Afficher l'aide")
        print("  quit - Quitter")
        print()

        while True:
            try:
                command = input("Teensy> ").strip()

                if command.lower() in ['quit', 'exit', 'q']:
                    break
                elif command.lower() == 'list':
                    files = self.list_files()
                    if files:
                        print("Fichiers disponibles:")
                        for f in files:
                            print(f"  - {f}")
                    else:
                        print("Aucun fichier trouvé")
                elif command.lower().startswith('get '):
                    filename = command[4:].strip()
                    if filename:
                        self.get_file(filename)
                    else:
                        print("Usage: get <filename>")
                elif command.lower() == 'help':
                    print(self.get_help())
                else:
                    print("Commande inconnue. Tapez 'help' pour voir les commandes disponibles.")

            except KeyboardInterrupt:
                print("\nArrêt...")
                break
            except Exception as e:
                print(f"Erreur: {e}")


def main():
    """Fonction principale."""
    parser = argparse.ArgumentParser(description="Interface Python pour Teensy SD")
    parser.add_argument('--port', '-p', help='Port série à utiliser')
    parser.add_argument('--baudrate', '-b', type=int, default=115200, help='Vitesse de communication')
    parser.add_argument('--list', '-l', action='store_true', help='Lister les fichiers et quitter')
    parser.add_argument('--get', '-g', help='Télécharger un fichier spécifique')
    parser.add_argument('--output-dir', '-o', default='downloaded_files', help='Répertoire de sortie')

    args = parser.parse_args()

    # Créer l'interface
    interface = TeensySDInterface(port=args.port, baudrate=args.baudrate)

    try:
        # Se connecter
        if not interface.connect():
            print("Impossible de se connecter au Teensy")
            return 1

        # Mode non-interactif
        if args.list:
            files = interface.list_files()
            for f in files:
                print(f)
        elif args.get:
            interface.get_file(args.get, args.output_dir)
        else:
            # Mode interactif
            interface.interactive_mode()

    except Exception as e:
        print(f"Erreur: {e}")
        return 1
    finally:
        interface.disconnect()

    return 0


if __name__ == "__main__":
    sys.exit(main())
