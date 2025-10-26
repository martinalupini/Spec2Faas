def print_purple(string):
    print("\033[95m"+string+"\033[0m")

def print_green(string):
    print("\033[92m"+string+"\033[0m")

def print_yellow(s): print("\033[93m {}\033[00m".format(s))

def print_blue(s): print("\033[96m {}\033[00m".format(s))

ROSA = '\033[95m'
RESET = '\033[0m'
BLUE = '\033[96m'

def dialogue(testo, interlocutore, larghezza=100):
    """
    Stampa un testo dentro una nuvoletta personalizzata con il nome all'interno
    e barre verticali.

    :param testo: La frase da mostrare.
    :param interlocutore: Il nome di chi parla (mostrato all'interno).
    :param larghezza: La larghezza massima del testo prima di andare a capo.
    """
    # Spezza il testo in più righe se necessario
    righe_testo = testo.splitlines()
    if not righe_testo:
        righe_testo = [""]

    # Calcola la larghezza massima necessaria per la nuvoletta,
    # considerando sia la lunghezza del nome sia quella del testo.
    lunghezza_nome = len(interlocutore)
    lunghezza_max_testo = max(len(riga) for riga in righe_testo)
    larghezza_interna = max(lunghezza_nome, lunghezza_max_testo)

    # Crea la linea di separazione
    separatore = "-" * larghezza_interna

    # --- Inizia a disegnare la nuvoletta ---

    # 1. Bordo superiore
    print("  " + "_" * (larghezza_interna + 2) + " ")

    # 2. Riga con il nome dell'interlocutore
    # Usiamo .ljust() per allineare e aggiungere spazi se necessario
    #nome_formattato = interlocutore.ljust(larghezza_interna)
    #print(" | " + nome_formattato + " |")
    nome_con_colore = f"{ROSA}{interlocutore}{RESET}"
    spazi_vuoti = " " * (larghezza_interna - lunghezza_nome)
    riga_nome = nome_con_colore + spazi_vuoti
    print(f" | {riga_nome} |")

    # 3. Linea di separazione
    print(" | " + separatore + " |")

    # 4. Righe del testo del dialogo
    for riga in righe_testo:
        # Applichiamo il colore blu solo al testo effettivo.
        # Gli spazi di riempimento vengono aggiunti dopo, per non
        # alterare l'allineamento.
        riga_con_colore = f"{BLUE}{riga}{RESET}"
        spazi_testo = " " * (larghezza_interna - len(riga))
        print(f" | {riga_con_colore}{spazi_testo} |")

    # 5. Bordo inferiore
    print(" '" + "_" * (larghezza_interna + 2) + "'")

    # 6. Coda (lingua) del fumetto a sinistra
    print(" /")
    print("/")