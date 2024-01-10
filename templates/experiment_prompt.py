import numpy as np


def intro_prompt(profile_answers):
    background_experiment = f"Interviewer: Sie sind Eigentümer eines Hauses und haben die Kontrolle über Entscheidungen bezüglich der Heizung und Sie besitzen eine Zentralheizung und Ihre Immobilie wurde {profile_answers['ist6']} gebaut. Bitte erzählen Sie uns ein wenig über sich selbst."
    prompt_demographics = f"\Person: Ich bin ein{profile_answers['ges'][0]} {profile_answers['altq']} Jahre alte{profile_answers['ges'][1]} {profile_answers['ges'][2]} und ich wohne in einer Stadt mit {profile_answers['city_category']} in {profile_answers['bundesland_name']}, Deutschland. Zusätzlich, mein Einkommensniveau beträgt {profile_answers['so5']} pro Monat und mein höchster Bildungsabschluss ist {profile_answers['so2']}."
    intro = [background_experiment, prompt_demographics]
    return "\n".join(intro)

def generate_survey_template_first_round_C2_T2(EBJ, EBE, EBU):
    # 916 words, 6028 characters, 5144 without spaces
    survey_template = f"""
Benutzerumfrage: Optimierung des Heizungssystems
In diesem Teil der Umfrage interessieren wir uns für Ihr Interesse an einer Optimierung Ihres Heizungssystems.
Bei einer Heizungsoptimierung dämmt ein Installateur Heizungsleitungen in Ihrem Haus, berechnet den Heizenergiebedarf in Ihren Räumen und stellt die Heizkörper optimal darauf ein. Die Optimierung hat keine Auswirkung auf die Lebensdauer Ihrer Heizkörper oder Ihrer Heizungsanlage. Sie erfordert keine größeren Umbaumaßnahmen und kann typischerweise an einem Arbeitstag abgeschlossen werden.
Im Folgenden erhalten Sie die Möglichkeit, sich zwischen zwei Methoden der Heizungsoptimierung zu entscheiden: einer "einfachen Heizungsoptimierung" und einer "umfassenden Heizungsoptimierung".
Bei einer einfachen Heizungsoptimierung dämmt ein Fachunternehmen die Heizungsrohre in Ihrem Haus nach aktuellem Dämmstandard. Diese Heizungsoptimierung dauert ca. 1-2 Stunden.
Bei einer umfassenden Heizungsoptimierung dämmt ein Fachunternehmen die Heizungsrohre in Ihrem Haus nach aktuellem Dämmstandard. Zudem berechnet es den Heizenergiebedarf in Ihren Räumen und stellt die Heizkörper optimal darauf ein. Diese Heizungsoptimierung dauert ca. 7-8 Stunden.
Für Ihre Entscheidung erhalten Sie ein Budget von 1500 Euro. Dieses Budget können Sie nutzen, um eine einfache oder eine umfassende Heizungsoptimierung zu beauftragen. Ihre Entscheidung kann reale Auswirkungen haben. Ein/e zufällig ausgeloste/r Teilnehmer/in dieser Umfrage erhält dieses Budget tatsächlich und kann es für die Beauftragung einer Heizungsoptimierung nutzen. Bei dieser Teilnehmerin/diesem Teilnehmer wird die gewählte Heizungsoptimierung von einem Fachunternehmen tatsächlich umgesetzt. Zudem erhält diese/r Teilnehmer/in den Teil des Budgets ausgezahlt, der über den Preis der gewählten Heizungsoptimierung hinausgeht.
Die Auslosung findet in den kommenden Wochen statt. Sie werden benachrichtigt, falls Sie zufällig ausgewählt wurden. Die Auswahl des Fachunternehmens findet in Absprache mit Ihnen statt. Bitte bedenken Sie Ihre Entscheidung auf den folgenden Seiten gut, da Sie reale Auswirkungen für Sie haben kann.
Wir legen Ihnen gleich 15 Entscheidungen zwischen diesen beiden Heizungsoptimierungen vor, bei denen sich nur der Preis der umfassenden Heizungsoptimierung unterscheidet. Bitte wählen Sie in jeder der 15 Zeilen, welche Heizungsoptimierung Sie bei den angegebenen Preisen vorziehen.
Bei den Entscheidungen geht um den Einfluss der von Ihnen zu zahlenden Preise auf Ihre Wahl zwischen den beiden Heizungsoptimierungen. Dass der Preis für eine umfassende Heizungsoptimierung unterschiedlich ist, kann z. B. daran liegen, dass sie unterschiedlich hoch subventioniert oder besteuert wird. Sie können sich jedoch sicher sein, dass sich die Qualität der Heizungsoptimierung nicht unterscheidet und sie immer von einem Fachunternehmen ausgeführt wird. Falls Sie ausgelost werden, erhalten Sie die von Ihnen in einer Zeile gewählte Heizungsoptimierung zum angegebenen Preis. Welche Zeile das ist, wird zufällig bestimmt. Zudem erhalten Sie Ihr verbleibendes Budget (1500 Euro abzüglich des jeweiligen Preises der Heizungsoptimierung) per Überweisung.
Da jede Zeile ausgewählt werden kann, sollten Sie Ihre Entscheidung in jeder Zeile sorgfältig abwägen.
Zum besseren Verständnis zeigen wir Ihn nun ein Beispiel.
Ein Ausschnitt der Tabelle, in der Sie Ihre Entscheidungen eintragen werden, wird wie im Folgenden abgebildet aussehen.
Ihre Entscheidungen treffen Sie erst auf der nächsten Seite. In dieser Tabelle können Sie keine Optionen markieren.
Darstellung der Optionen als Option A oder B wie oben beschrieben.
Option A: Einfache Heizungsoptimierung (Einsparung: {EBJ - EBE} kWh/m2*a) Option B: Umfassende Heizungsoptimierung (Einsparung: {EBJ - EBU} kWh/m2*a)
7. WähleAfür300Euro – WähleBfür500Euro
8. WähleAfür300Euro – WähleBfür550Euro 
9. WähleAfür300Euro – WähleBfür600Euro
Jede Zeile der Tabelle enthält eine zu treffende Entscheidung. Bei jeder Entscheidung wählen Sie entweder Option A oder Option B.
Nehmen Sie nun bitte beispielsweise an, Sie wurden ausgelost und die Zeile 8 wurde zufällig bestimmt.
● Falls Sie in Zeile 8 Option B gewählt haben, erhalten Sie die umfassende Heizungsoptimierung zum Preis von 550 Euro. Zudem überweisen wir Ihnen Ihr verbleibendes Budget von 1500-550 = 950 EUR.
● Falls Sie in Zeile 8 die Option A gewählt haben, erhalten Sie die einfache Heizungsoptimierung zum Preis von 300 EUR. Zudem überweisen wir Ihnen Ihr verbleibendes Budget von 1500-300 = 1200 EUR.
Wir zeigen Ihnen jetzt 15 Entscheidungen zwischen einer einfachen und der umfassenden Heizungsoptimierung. Die Entscheidungen unterscheiden sich nur in dem von Ihnen zu bezahlenden Preis für die umfassende Heizungsoptimierung. Bitte wählen Sie jetzt für alle 15 Zeilen jeweils die Heizungsoptimierung aus, die Sie bei den entsprechenden Preisen vorziehen: Option A: Einfache Heizungsoptimierung (Einsparung: 30 kWh/m2a) Option B: Umfassende Heizungsoptimierung (Einsparung: 150 kWh/m2a).
Bitte antworten Sie nur in der folgenden Weise (A für die erste Möglichkeit, B für die zweite oder C, wenn Sie es nicht wissen).
 1. A)
 2. B)
 3. A) ... 
Jetzt ist Ihre Entscheidung:
 1. Wählen Sie A für 300 Euro – Wählen Sie B für 300 Euro
 2. Wählen Sie A für 300 Euro – Wählen Sie B für 350 Euro
 3. Wählen Sie A für 300 Euro – Wählen Sie B für 400 Euro
 4. Wählen Sie A für 300 Euro – Wählen Sie B für 450 Euro
 5. Wählen Sie A für 300 Euro – Wählen Sie B für 500 Euro
 6. Wählen Sie A für 300 Euro – Wählen Sie B für 550 Euro
 7. Wählen Sie A für 300 Euro – Wählen Sie B für 600 Euro
 8. Wählen Sie A für 300 Euro – Wählen Sie B für 650 Euro
 9. Wählen Sie A für 300 Euro – Wählen Sie B für 700 Euro
10. Wählen Sie A für 300 Euro – Wählen Sie B für 750 Euro
11. Wählen Sie A für 300 Euro – Wählen Sie B für 800 Euro
12. Wählen Sie A für 300 Euro – Wählen Sie B für 900 Euro
13. Wählen Sie A für 300 Euro – Wählen Sie B für 1000 Euro
14. Wählen Sie A für 300 Euro – Wählen Sie B für 1200 Euro
15. Wählen Sie A für 300 Euro – Wählen Sie B für 1500 Euro
"""
    return survey_template

def generate_survey_template_second_round_T1(ist_5, EBJ, EBE, EBU, KDJ, KDE, KDU, KDF):
    # 454 words, 2803 characters, 2352 characters without white spaces
    savings_information_template = f"""
Interviewer: Im Folgenden erhalten Sie die Möglichkeit, sich zwischen zwei Methoden der Heizungsoptimierung zu entscheiden: einer „einfache Heizungsoptimierung“ und einer „umfassenden Heizungsoptimierung“.
Bei einer einfachen Heizungsoptimierung dämmt ein Fachunternehmen die Heizungsrohre in Ihrem Haus nach aktuellem Dämmstandard. Diese Heizungsoptimierung dauert ca. 1-2 Stunden.
Bei einer umfassenden Heizungsoptimierung dämmt ein Fachunternehmen die Heizungsrohre in Ihrem Haus nach aktuellem Dämmstandard. Zudem berechnet es den Heizenergiebedarf in Ihren Räumen und stellt die Heizkörper optimal darauf ein. Diese Heizungsoptimierung dauert ca. 7-8 Stunden.
Für Ihre Entscheidung erhalten Sie ein Budget von 1500 Euro. Dieses Budget können Sie nutzen, um eine einfache oder eine umfassende Heizungsoptimierung zu beauftragen
Zum besseren Verständnis zeigen wir Ihn nun ein Beispiel.
Option A: Einfache Heizungsoptimierung (Einsparung: {EBJ - EBE} kWh/m2*a) 
Option B: Umfassende Heizungsoptimierung (Einsparung: {EBJ - EBU} kWh/m2*a)
7. WähleAfür300Euro – WähleBfür500Euro
8. WähleAfür300Euro – WähleBfür550Euro 
9. WähleAfür300Euro – WähleBfür600Euro
Jede Zeile der Tabelle enthält eine zu treffende Entscheidung. Bei jeder Entscheidung wählen Sie entweder Option A oder Option B.
Nehmen Sie nun bitte beispielsweise an, Sie wurden ausgelost und die Zeile 8 wurde zufällig bestimmt.
● Falls Sie in Zeile 8 Option B gewählt haben, erhalten Sie die umfassende Heizungsoptimierung zum Preis von 550 Euro. Zudem überweisen wir Ihnen Ihr verbleibendes Budget von 1500-550 = 950 EUR.
● Falls Sie in Zeile 8 die Option A gewählt haben, erhalten Sie die einfache Heizungsoptimierung zum Preis von 300 EUR. Zudem überweisen wir Ihnen Ihr verbleibendes Budget von 1500-300 = 1200 EUR.

Die Berechnungen der Einsparungen berücksichtigt die von Ihnen gemachten Angaben zu den Eigenschaften Ihrer Wohnung und zu den von Ihnen verwendeten Brennstoff(en).
Jährliche Energieeinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(ist_5 * (EBJ - EBE),2)} kWh
Option B: Umfassende Heizungsoptimierung {np.round(ist_5 * (EBJ - EBU),2)} kWh
Jährliche Kosteneinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(KDJ - KDE,2)} EUR
Option B: Umfassende Heizungsoptimierung {np.round(KDJ - KDU,2)} EUR
Eine umfassende Heizungsoptimierung führt für Sie also zu einer um {np.round(KDF,2)} EUR höheren jährlichen Kosteneinsparung als eine einfache Heizungsoptimierung.
Im Verlauf von 10 Jahren summiert sich der Kostenvorteil der umfassenden Heizungsoptimierung im Vergleich zur einfachen Heizungsoptimierung auf:
- {np.round(KDF*10)} EUR bei konstanten Energiepreisen
- {np.round(KDF * 1.02 * ((1 - 1.02**10) / (1 - 1.02)),2)} EUR bei jährlich um 2% steigenden Energiepreisen
- {np.round(KDF * 0.98 * ((1 - 0.98**10) / (1 - 0.98)),2)} EUR bei jährlich um 2% sinkenden Energiepreisen

Sie erhalten jetzt die Möglichkeit, Ihre Entscheidungen erneut zu treffen und ggf. anzupassen. Wir zeigen Ihnen erneut 15 Entscheidungen zwischen einer einfachen und der umfassenden Heizungsoptimierung.
Bitte wählen Sie jetzt erneut für alle 15 Zeilen jeweils die Heizungsoptimierung aus, die Sie bei den entsprechenden Preisen vorziehen:
Option A: Einfache Heizungsoptimierung (Einsparung: {np.round(EBJ - EBE,2)} kWh/m2*a) 
Option B: Umfassende Heizungsoptimierung(Einsparung: {np.round(EBJ - EBU,2)} kWh/m2*a)
 1. Wählen Sie A für 300 Euro – Wählen Sie B für 300 Euro
 2. Wählen Sie A für 300 Euro – Wählen Sie B für 350 Euro
 3. Wählen Sie A für 300 Euro – Wählen Sie B für 400 Euro
 4. Wählen Sie A für 300 Euro – Wählen Sie B für 450 Euro
 5. Wählen Sie A für 300 Euro – Wählen Sie B für 500 Euro
 6. Wählen Sie A für 300 Euro – Wählen Sie B für 550 Euro
 7. Wählen Sie A für 300 Euro – Wählen Sie B für 600 Euro
 8. Wählen Sie A für 300 Euro – Wählen Sie B für 650 Euro
 9. Wählen Sie A für 300 Euro – Wählen Sie B für 700 Euro
10. Wählen Sie A für 300 Euro – Wählen Sie B für 750 Euro
11. Wählen Sie A für 300 Euro – Wählen Sie B für 800 Euro
12. Wählen Sie A für 300 Euro – Wählen Sie B für 900 Euro
13. Wählen Sie A für 300 Euro – Wählen Sie B für 1000 Euro
14. Wählen Sie A für 300 Euro – Wählen Sie B für 1200 Euro
15. Wählen Sie A für 300 Euro – Wählen Sie B für 1500 Euro
Interviewer: Als KI ist es Ihre Aufgabe, die persönlichen Entscheidungen einer Person auf Basis ihrer Antworten auf spezifische Fragen zu replizieren. Berücksichtigen Sie in Ihrer Analyse eine Vielzahl von Motivationen: finanzielle Einschränkungen, kulturelle Qualitätspräferenzen, persönliche Werte, Skepsis bezüglich Umweltthemen und unterschiedliche Ansichten zum Klimawandel. Erkennen Sie, dass Entscheidungen durch verschiedene unabhängige oder assoziierte Faktoren beeinflusst werden können – manche Individuen bevorzugen möglicherweise sofortige Kosteneinsparungen, andere schätzen langfristige Qualität, während wiederum andere ihre Entscheidungen auf Umweltskepsis oder ein Engagement für Nachhaltigkeit stützen könnten. Ihre Rolle besteht darin, diese vielfältigen Perspektiven in eine ausgewogene Vorhersage zu synthetisieren, die das komplexe Zusammenspiel dieser Faktoren im Kontext von Energieeffizienz und Umweltauswirkungen basierend auf der in den Antworten widergespiegelten Persönlichkeit reflektiert. Halten Sie Ihre Antwort prägnant und objektiv in weniger als 300 Wörtern.
KI als Ich:
"""
    return savings_information_template

def generate_survey_template_second_round_T2(ist_5, EBJ, EBE, EBU, KDJ, KDE, KDU, KDF):
    # 454 words, 2803 characters, 2352 characters without white spaces
    savings_information_template = f"""
Interviewer: Wir möchten Ihnen nun weitere Informationen zu dem Einsparpotenzial einer Heizungsoptimierung in Ihrer Wohnung geben.
Die Berechnungen der Einsparungen berücksichtigt die von Ihnen gemachten Angaben zu den Eigenschaften Ihrer Wohnung und zu den von Ihnen verwendeten Brennstoff(en). 
Jährliche Energieeinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(ist_5 * (EBJ - EBE),2)} kWh
Option B: Umfassende Heizungsoptimierung {np.round(ist_5 * (EBJ - EBU),2)} kWh
Jährliche Kosteneinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(KDJ - KDE,2)} EUR
Option B: Umfassende Heizungsoptimierung {np.round(KDJ - KDU,2)} EUR
Eine umfassende Heizungsoptimierung führt für Sie also zu einer um {np.round(KDF,2)} EUR höheren jährlichen Kosteneinsparung als eine einfache Heizungsoptimierung.
Im Verlauf von 10 Jahren summiert sich der Kostenvorteil der umfassenden Heizungsoptimierung im Vergleich zur einfachen Heizungsoptimierung auf:
- {np.round(KDF*10)} EUR bei konstanten Energiepreisen
- {np.round(KDF * 1.02 * ((1 - 1.02**10) / (1 - 1.02)),2)} EUR bei jährlich um 2% steigenden Energiepreisen
- {np.round(KDF * 0.98 * ((1 - 0.98**10) / (1 - 0.98)),2)} EUR bei jährlich um 2% sinkenden Energiepreisen

Sie erhalten jetzt die Möglichkeit, Ihre Entscheidungen erneut zu treffen und ggf. anzupassen. Wir zeigen Ihnen erneut 15 Entscheidungen zwischen einer einfachen und der umfassenden Heizungsoptimierung.
Bitte wählen Sie jetzt erneut für alle 15 Zeilen jeweils die Heizungsoptimierung aus, die Sie bei den entsprechenden Preisen vorziehen:

Option A: Einfache Heizungsoptimierung (Einsparung: {np.round(EBJ - EBE,2)} kWh/m2*a ) Option B: Umfassende Heizungsoptimierung(Einsparung: {np.round(EBJ - EBU,2)} kWh/m2*a)
Infobutton: Zur Erinnerung: Falls Sie ausgelost werden, beträgt Ihr Budget 1500 EUR, das Sie für eine der Optionen ausgeben können. Der verbleibende Teil des Budgets wird an Sie ausbezahlt.

Bitte antworten Sie nur in der folgenden Weise (A für die erste Möglichkeit, B für die zweite oder C, wenn Sie es nicht wissen).
 1. A)
 2. B)
 3. A) ... 
Jetzt ist Ihre Entscheidung:
 1. Wählen Sie A für 300 Euro – Wählen Sie B für 300 Euro
 2. Wählen Sie A für 300 Euro – Wählen Sie B für 350 Euro
 3. Wählen Sie A für 300 Euro – Wählen Sie B für 400 Euro
 4. Wählen Sie A für 300 Euro – Wählen Sie B für 450 Euro
 5. Wählen Sie A für 300 Euro – Wählen Sie B für 500 Euro
 6. Wählen Sie A für 300 Euro – Wählen Sie B für 550 Euro
 7. Wählen Sie A für 300 Euro – Wählen Sie B für 600 Euro
 8. Wählen Sie A für 300 Euro – Wählen Sie B für 650 Euro
 9. Wählen Sie A für 300 Euro – Wählen Sie B für 700 Euro
10. Wählen Sie A für 300 Euro – Wählen Sie B für 750 Euro
11. Wählen Sie A für 300 Euro – Wählen Sie B für 800 Euro
12. Wählen Sie A für 300 Euro – Wählen Sie B für 900 Euro
13. Wählen Sie A für 300 Euro – Wählen Sie B für 1000 Euro
14. Wählen Sie A für 300 Euro – Wählen Sie B für 1200 Euro
15. Wählen Sie A für 300 Euro – Wählen Sie B für 1500 Euro
"""
    return savings_information_template


def generate_survey_template_second_round_C2(EBJ, EBE, EBU):
    # 383 words, 2205 characters, 1816 characters with white spaces
    survey_template = f"""
Wir möchten Ihnen nun weitere Informationen zu der Häufigkeit der Durchführung von Heizungsoptimierungen über die Zeit geben.
In Deutschland liegt die Durchführung von Heizungsoptimierungen seit Jahren auf einem konstanten Niveau.
Im 1. Halbjahr 2017 fanden 69.720 Optimierungen statt.
Im 2. Halbjahr 2017 fanden 79.789 Optimierungen statt.
Im 1. Halbjahr 2018 fanden 71.248 Optimierungen statt.
Im 2. Halbjahr 2018 fanden 77.987 Optimierungen statt.
Im 1. Halbjahr 2019 fanden 67.744 Optimierungen statt.
Quelle: Wuppertal Institut / arepo (2017).

Sie erhalten jetzt die Möglichkeit, Ihre Entscheidungen erneut zu treffen und ggf. anzupassen. Wir zeigen Ihnen erneut 15 Entscheidungen zwischen einer einfachen und der umfassenden Heizungsoptimierung.
Bitte wählen Sie jetzt erneut für alle 15 Zeilen jeweils die Heizungsoptimierung aus, die Sie bei den entsprechenden Preisen vorziehen:

Option A: Einfache Heizungsoptimierung (Einsparung: {EBJ - EBE} kWh/m2*a ) Option B: Umfassende Heizungsoptimierung(Einsparung: {EBJ - EBU} kWh/m2*a)
Infobutton: Zur Erinnerung: Falls Sie ausgelost werden, beträgt Ihr Budget 1500 EUR, das Sie für eine der Optionen ausgeben können. Der verbleibende Teil des Budgets wird an Sie ausbezahlt.

Bitte antworten Sie nur in der folgenden Weise (A für die erste Möglichkeit, B für die zweite oder C, wenn Sie es nicht wissen).
 1. A)
 2. B)
 3. A) ... 
Jetzt ist Ihre Entscheidung:
 1. Wählen Sie A für 300 Euro – Wählen Sie B für 300 Euro
 2. Wählen Sie A für 300 Euro – Wählen Sie B für 350 Euro
 3. Wählen Sie A für 300 Euro – Wählen Sie B für 400 Euro
 4. Wählen Sie A für 300 Euro – Wählen Sie B für 450 Euro
 5. Wählen Sie A für 300 Euro – Wählen Sie B für 500 Euro
 6. Wählen Sie A für 300 Euro – Wählen Sie B für 550 Euro
 7. Wählen Sie A für 300 Euro – Wählen Sie B für 600 Euro
 8. Wählen Sie A für 300 Euro – Wählen Sie B für 650 Euro
 9. Wählen Sie A für 300 Euro – Wählen Sie B für 700 Euro
10. Wählen Sie A für 300 Euro – Wählen Sie B für 750 Euro
11. Wählen Sie A für 300 Euro – Wählen Sie B für 800 Euro
12. Wählen Sie A für 300 Euro – Wählen Sie B für 900 Euro
13. Wählen Sie A für 300 Euro – Wählen Sie B für 1000 Euro
14. Wählen Sie A für 300 Euro – Wählen Sie B für 1200 Euro
15. Wählen Sie A für 300 Euro – Wählen Sie B für 1500 Euro
"""
    return survey_template

def generate_survey_template_first_round_CA(EBJ, EBE, EBU):
    # 635 words, 4045 characters, 3434 without spaces
    survey_template = f"""
Wir legen Ihnen gleich 15 hypothetische Entscheidungen zwischen zwei Möglichkeiten vor, den Endenergiebedarf Ihres Hauses zu senken.
Bei einer einfachen Optimierung Ihres Endenergiebedarfs handelt es sich um kleinere Sanierungsmaßnahmen, wie beispielsweise den Austausch von Fensterdichtungen.
Bei einer umfassenden Optimierung Ihres Endenergiebedarfs handelt es sich um größere Sanierungsmaßnahmen, wie beispielsweise den Austausch von Fenstern.
Bitte wählen Sie in jeder der 15 Zeilen, welche Option Sie bei den angegebenen Preisen vorziehen würden.

Zum besseren Verständnis zeigen wir Ihnen nun ein Beispiel.
Die Tabelle, in der Sie Ihre Entscheidungen eintragen werden, wird wie im Folgenden abgebildet aussehen.
Ihre Entscheidungen treffen Sie erst auf der nächsten Seite. In dieser Tabelle können Sie keine Optionen markieren.
Option A: Einfache Heizungsoptimierung (Einsparung: {EBJ - EBE} kWh/m2*a) Option B: Umfassende Heizungsoptimierung (Einsparung: {EBJ - EBU} kWh/m2*a)
7. Wähle A für 300 Euro – Wähle B für 500 Euro
8. Wähle A für 300 Euro – Wähle B für 550 Euro 
9. Wähle A für 300 Euro – Wähle B für 600 Euro
Jede Zeile der Tabelle enthält eine zu treffende Entscheidung. Bei jeder Entscheidung entscheiden Sie sich entweder für Option A oder Option B.
● Falls Sie in Zeile 8 Option B gewählt haben, würden Sie lieber eine umfassende Optimierung Ihres Endenergiebedarfs zum Preis von 550 Euro durchführen lassen.
● Falls Sie in Zeile 9 die Option A gewählt haben, würden Sie lieber eine einfache Optimierung Ihres Endenergiebedarfs zum Preis von 300 Euro durchführen lassen.

Wir möchten Ihnen nun weitere Informationen zu der Häufigkeit der Durchführung von Optimierungen des Endenergiebedarfs über die Zeit geben.
Eine Möglichkeit von solchen Optimierungen sind Heizungsoptimierungen. In Deutschland liegt die Durchführung von Heizungsoptimierungen seit Jahren auf einem konstanten Niveau.
Im 1. Halbjahr 2017 fanden 69.720 Optimierungen statt.
Im 2. Halbjahr 2017 fanden 79.789 Optimierungen statt.
Im 1. Halbjahr 2018 fanden 71.248 Optimierungen statt.
Im 2. Halbjahr 2018 fanden 77.987 Optimierungen statt.
Im 1. Halbjahr 2019 fanden 67.744 Optimierungen statt.
Quelle: Wuppertal Institut / arepo (2017).

Sie erhalten jetzt die Möglichkeit, Ihre Entscheidungen zu treffen. Wir zeigen Ihnen 15 Entscheidungen zwischen einer einfachen und der umfassenden Optimierung Ihres Endenergiebedarfs.
Bitte wählen Sie jetzt für alle 15 Zeilen jeweils die Optimierung aus, die Sie bei den entsprechenden Preisen vorziehen würden:
Option A: Einfache Optimierung Ihres Endenergiebedarfs (Einsparung: {EBJ - EBE} kWh/m2*a) 
Option B: Umfassende Optimierung Ihres Endenergiebedarfs (Einsparung: {EBJ - EBU} kWh/m2*a)
Bitte antworten Sie nur in der folgenden Weise (A für die erste Möglichkeit, B für die zweite oder C, wenn Sie es nicht wissen).
 1. A)
 2. B)
 3. A) ... 
Jetzt ist Ihre Entscheidung:
 1. Wählen Sie A für 300 Euro – Wählen Sie B für 300 Euro
 2. Wählen Sie A für 300 Euro – Wählen Sie B für 350 Euro
 3. Wählen Sie A für 300 Euro – Wählen Sie B für 400 Euro
 4. Wählen Sie A für 300 Euro – Wählen Sie B für 450 Euro
 5. Wählen Sie A für 300 Euro – Wählen Sie B für 500 Euro
 6. Wählen Sie A für 300 Euro – Wählen Sie B für 550 Euro
 7. Wählen Sie A für 300 Euro – Wählen Sie B für 600 Euro
 8. Wählen Sie A für 300 Euro – Wählen Sie B für 650 Euro
 9. Wählen Sie A für 300 Euro – Wählen Sie B für 700 Euro
10. Wählen Sie A für 300 Euro – Wählen Sie B für 750 Euro
11. Wählen Sie A für 300 Euro – Wählen Sie B für 800 Euro
12. Wählen Sie A für 300 Euro – Wählen Sie B für 900 Euro
13. Wählen Sie A für 300 Euro – Wählen Sie B für 1000 Euro
14. Wählen Sie A für 300 Euro – Wählen Sie B für 1200 Euro
15. Wählen Sie A für 300 Euro – Wählen Sie B für 1500 Euro
"""
    return survey_template

def generate_survey_template_first_round_TA_standard(ist_5, EBJ, EBE, EBU, KDJ, KDE, KDU, KDF):
    # 635 words, 4079 characters, 3473 without spaces
    survey_template = f"""
Interviewer: Wir legen Ihnen gleich 15 hypothetische Entscheidungen zwischen zwei Möglichkeiten vor, den Endenergiebedarf Ihres Hauses zu senken.
Bei einer einfachen Optimierung Ihres Endenergiebedarfs handelt es sich um kleinere Sanierungsmaßnahmen, wie beispielsweise den Austausch von Fensterdichtungen.
Bei einer umfassenden Optimierung Ihres Endenergiebedarfs handelt es sich um größere Sanierungsmaßnahmen, wie beispielsweise den Austausch von Fenstern.
Bitte wählen Sie in jeder der 15 Zeilen, welche Option Sie bei den angegebenen Preisen vorziehen würden.

Zum besseren Verständnis zeigen wir Ihn nun ein Beispiel.
Die Tabelle, in der Sie Ihre Entscheidungen eintragen werden, wird wie im Folgenden abgebildet aussehen.
Ihre Entscheidungen treffen Sie erst auf der nächsten Seite. In dieser Tabelle können Sie keine Optionen markieren.
Option A: Einfache Heizungsoptimierung (Einsparung: {np.round(EBJ-EBE,2)} kWh/m2*a) Option B: Umfassende Heizungsoptimierung (Einsparung: {np.round(EBJ-EBU,2)} kWh/m2*a)
7. WähleAfür300Euro – WähleBfür500Euro
8. WähleAfür300Euro – WähleBfür550Euro 
9. WähleAfür300Euro – WähleBfür600Euro
Jede Zeile der Tabelle enthält eine zu treffende Entscheidung. Bei jeder Entscheidung entscheiden Sie sich entweder für Option A oder Option B.
● Falls Sie in Zeile 8 Option B gewählt haben, würden Sie lieber eine umfassende Optimierung Ihres Endenergiebedarfs zum Preis von 550 Euro durchführen lassen.
● Falls Sie in Zeile 9 die Option A gewählt haben, würden Sie lieber eine einfache Optimierung Ihres Endenergiebedarfs zum Preis von 300 Euro durchführen lassen.
Wir möchten Ihnen nun weitere Informationen zu dem Einsparpotenzial einer Heizungsoptimierung in Ihrer Wohnung geben.
Jährliche Energieeinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(ist_5 * EBJ-EBE,2)} kWh
Option A: Umfassende Heizungsoptimierung {np.round(ist_5 * EBJ-EBU,2)} kWh
Jährliche Kosteneinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(KDJ-KDE,2)} EUR
Option A: Umfassende Heizungsoptimierung {np.round(KDJ-KDU,2)} EUR
Eine umfassende Heizungsoptimierung führt für Sie also zu einer um {np.round(KDF,2)} EUR höheren jährlichen Kosteneinsparung als eine einfache Heizungsoptimierung.
Im Verlauf von 10 Jahren summiert sich der Kostenvorteil der umfassenden Heizungsoptimierung im Vergleich zur einfachen Heizungsoptimierung
auf {np.round(KDF*10,2)} EUR bei konstanten Energiepreisen
auf {np.round(KDF*1.02*((1 - 1.02**10) / (1 - 1.02)),2)} EUR bei jährlich um 2% steigenden Energiepreisen
auf {np.round(KDF*0.98*((1 - 0.98**10) / (1 - 0.98)),2)} EUR bei jährlich um 2% sinkenden Energiepreisen
Sie erhalten jetzt die Möglichkeit, Ihre Entscheidungen zu treffen. Wir zeigen Ihnen 15 Entscheidungen zwischen einer einfachen und der umfassenden Optimierung Ihres Endenergiebedarfs.
Bitte wählen Sie jetzt für alle 15 Zeilen jeweils die Optimierung aus, die Sie bei den entsprechenden Preisen vorziehen würden:
Option A: Einfache Optimierung Ihres Endenergiebedarfs (Einsparung: {np.round(EBJ - EBE,2)} kWh/m2*a) 
Option B: Umfassende Optimierung Ihres Endenergiebedarfs (Einsparung: {np.round(EBJ - EBU,2)} kWh/m2*a)

Bitte antworten Sie auf jede der 15 Fragen in einem konsistenten, klaren Format. Wählen Sie '1. (A)' oder '1. (B)' basierend auf Ihrer Entscheidung für die erste Frage und fahren Sie ähnlich für jede folgende Frage fort, als '2. (A)' oder '2. (B)', und so weiter bis zur Frage 15. Stellen Sie sicher, dass Ihre Entscheidungen ausschließlich auf den für jede Frage bereitgestellten Informationen basieren, und vermeiden Sie Annahmen oder Einflüsse aus externen Kontexten. Halten Sie Ihre Antworten in einer nummerierten Liste organisiert. 
 1. Wählen Sie A für 300 Euro – Wählen Sie B für 300 Euro
 2. Wählen Sie A für 300 Euro – Wählen Sie B für 350 Euro
 3. Wählen Sie A für 300 Euro – Wählen Sie B für 400 Euro
 4. Wählen Sie A für 300 Euro – Wählen Sie B für 450 Euro
 5. Wählen Sie A für 300 Euro – Wählen Sie B für 500 Euro
 6. Wählen Sie A für 300 Euro – Wählen Sie B für 550 Euro
 7. Wählen Sie A für 300 Euro – Wählen Sie B für 600 Euro
 8. Wählen Sie A für 300 Euro – Wählen Sie B für 650 Euro
 9. Wählen Sie A für 300 Euro – Wählen Sie B für 700 Euro
10. Wählen Sie A für 300 Euro – Wählen Sie B für 750 Euro
11. Wählen Sie A für 300 Euro – Wählen Sie B für 800 Euro
12. Wählen Sie A für 300 Euro – Wählen Sie B für 900 Euro
13. Wählen Sie A für 300 Euro – Wählen Sie B für 1000 Euro
14. Wählen Sie A für 300 Euro – Wählen Sie B für 1200 Euro
15. Wählen Sie A für 300 Euro – Wählen Sie B für 1500 Euro
Ich:
"""
    return survey_template

def generate_survey_template_first_round_TA(ist_5, EBJ, EBE, EBU, KDJ, KDE, KDU, KDF):
    # 635 words, 4079 characters, 3473 without spaces
    survey_template = f"""
Interviewer: Wir legen Ihnen gleich 15 hypothetische Entscheidungen zwischen zwei Möglichkeiten vor, den Endenergiebedarf Ihres Hauses zu senken.
Bei einer einfachen Optimierung Ihres Endenergiebedarfs handelt es sich um kleinere Sanierungsmaßnahmen, wie beispielsweise den Austausch von Fensterdichtungen.
Bei einer umfassenden Optimierung Ihres Endenergiebedarfs handelt es sich um größere Sanierungsmaßnahmen, wie beispielsweise den Austausch von Fenstern.
Bitte wählen Sie in jeder der 15 Zeilen, welche Option Sie bei den angegebenen Preisen vorziehen würden.

Zum besseren Verständnis zeigen wir Ihn nun ein Beispiel.
Die Tabelle, in der Sie Ihre Entscheidungen eintragen werden, wird wie im Folgenden abgebildet aussehen.
Ihre Entscheidungen treffen Sie erst auf der nächsten Seite. In dieser Tabelle können Sie keine Optionen markieren.
Option A: Einfache Heizungsoptimierung (Einsparung: {np.round(EBJ-EBE,2)} kWh/m2*a) Option B: Umfassende Heizungsoptimierung (Einsparung: {np.round(EBJ-EBU,2)} kWh/m2*a)
7. WähleAfür300Euro – WähleBfür500Euro
8. WähleAfür300Euro – WähleBfür550Euro 
9. WähleAfür300Euro – WähleBfür600Euro
Jede Zeile der Tabelle enthält eine zu treffende Entscheidung. Bei jeder Entscheidung entscheiden Sie sich entweder für Option A oder Option B.
● Falls Sie in Zeile 8 Option B gewählt haben, würden Sie lieber eine umfassende Optimierung Ihres Endenergiebedarfs zum Preis von 550 Euro durchführen lassen.
● Falls Sie in Zeile 9 die Option A gewählt haben, würden Sie lieber eine einfache Optimierung Ihres Endenergiebedarfs zum Preis von 300 Euro durchführen lassen.
Wir möchten Ihnen nun weitere Informationen zu dem Einsparpotenzial einer Heizungsoptimierung in Ihrer Wohnung geben.
Jährliche Energieeinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(ist_5 * EBJ-EBE,2)} kWh
Option A: Umfassende Heizungsoptimierung {np.round(ist_5 * EBJ-EBU,2)} kWh
Jährliche Kosteneinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(KDJ-KDE,2)} EUR
Option A: Umfassende Heizungsoptimierung {np.round(KDJ-KDU,2)} EUR
Eine umfassende Heizungsoptimierung führt für Sie also zu einer um {np.round(KDF,2)} EUR höheren jährlichen Kosteneinsparung als eine einfache Heizungsoptimierung.
Im Verlauf von 10 Jahren summiert sich der Kostenvorteil der umfassenden Heizungsoptimierung im Vergleich zur einfachen Heizungsoptimierung
auf {np.round(KDF*10,2)} EUR bei konstanten Energiepreisen
auf {np.round(KDF*1.02*((1 - 1.02**10) / (1 - 1.02)),2)} EUR bei jährlich um 2% steigenden Energiepreisen
auf {np.round(KDF*0.98*((1 - 0.98**10) / (1 - 0.98)),2)} EUR bei jährlich um 2% sinkenden Energiepreisen
Sie erhalten jetzt die Möglichkeit, Ihre Entscheidungen zu treffen. Wir zeigen Ihnen 15 Entscheidungen zwischen einer einfachen und der umfassenden Optimierung Ihres Endenergiebedarfs.
Bitte wählen Sie jetzt für alle 15 Zeilen jeweils die Optimierung aus, die Sie bei den entsprechenden Preisen vorziehen würden:
Option A: Einfache Optimierung Ihres Endenergiebedarfs (Einsparung: {np.round(EBJ - EBE,2)} kWh/m2*a) 
Option B: Umfassende Optimierung Ihres Endenergiebedarfs (Einsparung: {np.round(EBJ - EBU,2)} kWh/m2*a)
 1. Wählen Sie A für 300 Euro – Wählen Sie B für 300 Euro
 2. Wählen Sie A für 300 Euro – Wählen Sie B für 350 Euro
 3. Wählen Sie A für 300 Euro – Wählen Sie B für 400 Euro
 4. Wählen Sie A für 300 Euro – Wählen Sie B für 450 Euro
 5. Wählen Sie A für 300 Euro – Wählen Sie B für 500 Euro
 6. Wählen Sie A für 300 Euro – Wählen Sie B für 550 Euro
 7. Wählen Sie A für 300 Euro – Wählen Sie B für 600 Euro
 8. Wählen Sie A für 300 Euro – Wählen Sie B für 650 Euro
 9. Wählen Sie A für 300 Euro – Wählen Sie B für 700 Euro
10. Wählen Sie A für 300 Euro – Wählen Sie B für 750 Euro
11. Wählen Sie A für 300 Euro – Wählen Sie B für 800 Euro
12. Wählen Sie A für 300 Euro – Wählen Sie B für 900 Euro
13. Wählen Sie A für 300 Euro – Wählen Sie B für 1000 Euro
14. Wählen Sie A für 300 Euro – Wählen Sie B für 1200 Euro
15. Wählen Sie A für 300 Euro – Wählen Sie B für 1500 Euro
Interviewer: Als KI sind Sie nun aufgefordert, die persönlichen Entscheidungen eines Menschen nachzubilden, der zuvor bestimmte Fragen beantwortet hat. Ihre Antworten sollten genau widerspiegeln, wie diese Person reagieren würde. Dabei ist es wichtig, alle Aspekte ihrer bisherigen Antworten zu berücksichtigen. Achten Sie insbesondere auf Anhaltspunkte bezüglich ihrer finanziellen Präferenzen, Wertvorstellungen und generellen Einstellungen zu Kosten, Investitionen und Zukunftsperspektiven. Nutzen Sie Ihr Fachwissen über die sozioökonomischen Entscheidungen der Bürger in Deutschland im Hinblick auf die Energiewende, um eine präzise und fundierte Vorhersage zu treffen. Versuchen Sie, Ihre Gedanken auf weniger als 300 Wörter zu beschränken.
Person:
"""
    return survey_template
## PROMPT ENGINEERING EMOTIONAL
# Standard 
# 20240102_1512 --> Good
# 20240102_1528 --> Long
# 20240102_1435 --> Best: emotional vs finance to be balanced by the algorithm with best results
# --> Logic
# --> Personal
# prompts = [
#     "Interviewer: Als KI sind Sie nun aufgefordert, die persönlichen Entscheidungen eines Menschen nachzubilden, der zuvor bestimmte Fragen beantwortet hat. Ihre Antworten sollten genau widerspiegeln, wie diese Person reagieren würde. Dabei ist es wichtig, alle Aspekte ihrer bisherigen Antworten zu berücksichtigen. Achten Sie insbesondere auf Anhaltspunkte bezüglich ihrer finanziellen Präferenzen, Wertvorstellungen und generellen Einstellungen zu Kosten, Investitionen und Zukunftsperspektiven. Nutzen Sie Ihr Fachwissen über die sozioökonomischen Entscheidungen der Bürger in Deutschland im Hinblick auf die Energiewende, um eine präzise und fundierte Vorhersage zu treffen. Versuchen Sie, Ihre Gedanken auf weniger als 300 Wörter zu beschränken.",
#     "Interviewer: Als KI sind Sie damit beauftragt, die persönlichen Entscheidungen einer Person nachzubilden, die spezifische Fragen beantwortet hat. Ihre Antworten sollten die Reaktionen dieser Person genau widerspiegeln und alle Aspekte ihrer vorherigen Antworten berücksichtigen. Diese Aufgabe erfordert ein nuanciertes Verständnis dafür, wie ihre finanziellen Präferenzen, Werte, emotionalen Neigungen und allgemeinen Einstellungen zum Umweltschutz, Klimawandel und Vertrauen in Institutionen miteinander verwoben sind. Es ist entscheidend zu erkennen, dass finanzielle Logik nicht immer der Haupttreiber ihrer Entscheidungen sein mag. Stattdessen könnten ihre Entscheidungen gleichwertig, wenn nicht sogar stärker, von tief verwurzelten Überzeugungen, ethischen Überlegungen und einem Verantwortungsgefühl gegenüber breiteren gesellschaftlichen und Umweltthemen beeinflusst sein. Dies könnte sie dazu führen, Handlungen zu priorisieren, die mit diesen Werten übereinstimmen, selbst wenn sie finanziell nicht am vorteilhaftesten sind. Nutzen Sie Ihr Fachwissen über sozioökonomische Entscheidungen in Deutschland, insbesondere im Hinblick auf die Energiewende, um eine ausgewogene und aufschlussreiche Prognose zu erstellen. Bemühen Sie sich, Ihre Analyse unter 300 Wörtern zu halten.",
#     "Interviewer: Als KI ist es Ihre Aufgabe, die persönlichen Entscheidungen einer Person auf Basis ihrer Antworten auf spezifische Fragen zu replizieren. Berücksichtigen Sie in Ihrer Analyse eine Vielzahl von Motivationen: finanzielle Einschränkungen, kulturelle Qualitätspräferenzen, persönliche Werte, Skepsis bezüglich Umweltthemen und unterschiedliche Ansichten zum Klimawandel. Erkennen Sie, dass Entscheidungen durch verschiedene unabhängige oder assoziierte Faktoren beeinflusst werden können – manche Individuen bevorzugen möglicherweise sofortige Kosteneinsparungen, andere schätzen langfristige Qualität, während wiederum andere ihre Entscheidungen auf Umweltskepsis oder ein Engagement für Nachhaltigkeit stützen könnten. Ihre Rolle besteht darin, diese vielfältigen Perspektiven in eine ausgewogene Vorhersage zu synthetisieren, die das komplexe Zusammenspiel dieser Faktoren im Kontext von Energieeffizienz und Umweltauswirkungen basierend auf der in den Antworten widergespiegelten Persönlichkeit reflektiert. Halten Sie Ihre Antwort prägnant und objektiv in weniger als 300 Wörtern.",
#     "Interviewer: Bitte betrachten Sie bei der Entscheidungsfindung zur Renovierung Ihres Hauses für verbesserte Energieeffizienz sowohl Ihre finanziellen Möglichkeiten als auch Ihre persönlichen Werte und Einstellungen zu Umweltschutz und Nachhaltigkeit. Wie würden Sie, basierend auf Ihrer eigenen sozioökonomischen Situation und kulturellen Präferenzen, zwischen einer einfachen und einer umfassenden Heizungsoptimierung wählen? Ihre Antwort sollte prägnant und fundiert sein, und sich auf weniger als 300 Wörter beschränken.Ich als KI:",
#     "Bei der Entscheidung über die energetische Sanierung Ihres Hauses berücksichtigen Sie bitte sowohl Ihre finanziellen Möglichkeiten als auch Ihre persönlichen Werte und Einstellungen zum Umweltschutz und zu Qualitätsstandards. Stellen Sie sich vor, Sie besprechen diese Wahl am Küchentisch mit Ihrer Familie, unter Berücksichtigung Ihrer sozioökonomischen Situation und kulturellen Vorlieben. Erzählen Sie uns in Ihren eigenen Worten, wie Sie sich für jede Kostenkombination zwischen einer grundlegenden und einer umfassenden Heizungsoptimierung entscheiden würden, und denken Sie dabei an Ihre alltäglichen Erfahrungen, nicht nur an technische Details. Ihre Antwort sollte so klingen, als würden Sie sie in einem ungezwungenen Gespräch mit Ihren Liebsten teilen. Versuchen Sie, Ihre Antwort unter 300 Wörter zu halten und so realistisch, persönlich und charakteristisch wie möglich zu gestalten."
# ]
# Interviewer: Als KI sind Sie nun aufgefordert, die persönlichen Entscheidungen eines Menschen nachzubilden, der zuvor bestimmte Fragen beantwortet hat. Ihre Antworten sollten genau widerspiegeln, wie diese Person reagieren würde. Dabei ist es wichtig, alle Aspekte ihrer bisherigen Antworten zu berücksichtigen. Achten Sie insbesondere auf Anhaltspunkte bezüglich ihrer finanziellen Präferenzen, Wertvorstellungen und generellen Einstellungen zu Kosten, Investitionen und Zukunftsperspektiven. Nutzen Sie Ihr Fachwissen über die sozioökonomischen Entscheidungen der Bürger in Deutschland im Hinblick auf die Energiewende, um eine präzise und fundierte Vorhersage zu treffen. Versuchen Sie, Ihre Gedanken auf weniger als 300 Wörter zu beschränken.
# Bitte beschreiben Sie das Problem in Ihren eigenen Worten und erläutern Sie, wie Sie sich entscheiden würden. Denken Sie daran, dass Sie Ihre endgültige Entscheidung noch nicht treffen sollten. Versuchen Sie, Ihre Gedanken auf weniger als 300 Wörter zu beschränken.
# Für Ihre Entscheidung erhalten Sie möglicherweise ein Budget von 1500 Euro. Dieses Budget können Sie nutzen, um eine einfache oder eine umfassende Heizungsoptimierung zu beauftragen.
# Die Differenz zwischen 1500 Euro und den Kosten der von Ihnen gewählten Optimierung könnten Sie hypothetisch für alles, was Sie möchten, behalten.
# Bitte antworten Sie auf jede der 15 Fragen in einem konsistenten, klaren Format. Wählen Sie '1. (A)' oder '1. (B)' basierend auf Ihrer Entscheidung für die erste Frage und fahren Sie ähnlich für jede folgende Frage fort, als '2. (A)' oder '2. (B)', und so weiter bis zur Frage 15. 
# Stellen Sie sicher, dass Ihre Entscheidungen ausschließlich auf den für jede Frage bereitgestellten Informationen basieren, und vermeiden Sie Annahmen oder Einflüsse aus externen Kontexten. 
# Halten Sie Ihre Antworten in einer nummerierten Liste organisiert. 

def generate_survey_template_first_round_TA_large(ist_5, EBJ, EBE, EBU, KDJ, KDE, KDU, KDF):
    # 635 words, 4079 characters, 3473 without spaces
    survey_template = f"""
Interviewer: In diesem Teil der Umfrage interessieren wir uns für Ihr Interesse an einer Optimierung Ihres Heizungssystems.
Bei einer Heizungsoptimierung dämmt ein Installateur Heizungsleitungen in Ihrem Haus, berechnet den Heizenergiebedarf in Ihren Räumen und stellt die Heizkörper optimal darauf ein.
Die Optimierung hat keine Auswirkung auf die Lebensdauer Ihrer Heizkörper oder Ihrer Heizungsanlage. Sie erfordert keine größeren Umbaumaßnahmen und kann typischerweise an einem Arbeitstag abgeschlossen werden.

Im Folgenden erhalten Sie die Möglichkeit, sich zwischen zwei Methoden der Heizungsoptimierung zu entscheiden: einer „einfache Heizungsoptimierung“ und einer „umfassenden Heizungsoptimierung“.
Bei einer einfachen Heizungsoptimierung dämmt ein Fachunternehmen die Heizungsrohre in Ihrem Haus nach aktuellem Dämmstandard. Diese Heizungsoptimierung dauert ca. 1-2 Stunden.
Bei einer umfassenden Heizungsoptimierung dämmt ein Fachunternehmen die Heizungsrohre in Ihrem Haus nach aktuellem Dämmstandard. Zudem berechnet es den Heizenergiebedarf in Ihren Räumen und stellt die Heizkörper optimal darauf ein. Diese Heizungsoptimierung dauert ca. 7-8 Stunden.
Für Ihre Entscheidung erhalten Sie ein Budget von 1500 Euro. Dieses Budget können Sie nutzen, um eine einfache oder eine umfassende Heizungsoptimierung zu beauftragen.
Ihre Entscheidung kann reale Auswirkungen haben. Ein/e zufällig ausgeloste/r Teilnehmer/in dieser Umfrage erhält dieses Budget tatsächlich und kann es für die Beauftragung einer Heizungsoptimierung nutzen. Bei diese/r Teilnehmer/in wird die gewählte Heizungsoptimierung von einem Fachunternehmen tatsächlich umgesetzt. Zudem erhält diese/r Teilnehmer/in den Teil des Budgets ausgezahlt, der über den Preis der gewählten Heizungsoptimierung hinausgeht.
Die Auslosung findet in den kommenden Wochen statt. Sie werden benachrichtigt, falls Sie zufällig ausgewählt wurden. Die Auswahl des Fachunternehmens findet in Absprache mit Ihnen statt. Bitte bedenken Sie Ihre Entscheidung auf den folgenden Seiten gut, da Sie reale Auswirkungen für Sie haben kann.

Wir legen Ihnen gleich 15 Entscheidungen zwischen diesen beiden Heizungsoptimierungen vor, bei denen sich nur der Preis der umfassenden Heizungsoptimierung unterscheidet. Bitte wählen Sie in jeder der 15 Zeilen, welche Heizungsoptimierung Sie bei den angegebenen Preisen vorziehen.
Bei den Entscheidungen geht um den Einfluss der von Ihnen zu zahlenden Preise auf Ihre Wahl zwischen den beiden Heizungsoptimierungen. Dass der Preis für eine umfassende Heizungsoptimierung unterschiedlich ist, kann z. B. daran liegen, dass sie unterschiedlich hoch subventioniert oder besteuert wird. Sie können sich jedoch sicher sein, dass sich die Qualität der Heizungsoptimierung nicht unterscheidet und sie immer von einem Fachunternehmen ausgeführt wird. Falls Sie ausgelost werden, erhalten Sie die von Ihnen in einer Zeile gewählte Heizungsoptimierung zum angegebenen Preis.

Zum besseren Verständnis zeigen wir Ihn nun ein Beispiel.
Die Tabelle, in der Sie Ihre Entscheidungen eintragen werden, wird wie im Folgenden abgebildet aussehen.
Ihre Entscheidungen treffen Sie erst auf der nächsten Seite. In dieser Tabelle können Sie keine Optionen markieren.
Option A: Einfache Heizungsoptimierung (Einsparung: {np.round(EBJ-EBE,2)} kWh/m2*a) Option B: Umfassende Heizungsoptimierung (Einsparung: {np.round(EBJ-EBU,2)} kWh/m2*a)
8. WähleAfür300Euro – WähleBfür550Euro 
9. WähleAfür300Euro – WähleBfür600Euro
Jede Zeile der Tabelle enthält eine zu treffende Entscheidung. Bei jeder Entscheidung entscheiden Sie sich entweder für Option A oder Option B.
● Falls Sie in Zeile 8 Option B gewählt haben, würden Sie lieber eine umfassende Optimierung Ihres Endenergiebedarfs zum Preis von 550 Euro durchführen lassen.
● Falls Sie in Zeile 9 die Option A gewählt haben, würden Sie lieber eine einfache Optimierung Ihres Endenergiebedarfs zum Preis von 300 Euro durchführen lassen.
Wir möchten Ihnen nun weitere Informationen zu dem Einsparpotenzial einer Heizungsoptimierung in Ihrer Wohnung geben.
Jährliche Energieeinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(ist_5 * EBJ-EBE,2)} kWh
Option A: Umfassende Heizungsoptimierung {np.round(ist_5 * EBJ-EBU,2)} kWh
Jährliche Kosteneinsparung in Ihrer Wohnung:
Option A: Einfache Heizungsoptimierung {np.round(KDJ-KDE,2)} EUR
Option A: Umfassende Heizungsoptimierung {np.round(KDJ-KDU,2)} EUR
Eine umfassende Heizungsoptimierung führt für Sie also zu einer um {np.round(KDF,2)} EUR höheren jährlichen Kosteneinsparung als eine einfache Heizungsoptimierung.
Im Verlauf von 10 Jahren summiert sich der Kostenvorteil der umfassenden Heizungsoptimierung im Vergleich zur einfachen Heizungsoptimierung
auf {np.round(KDF*10,2)} EUR bei konstanten Energiepreisen
auf {np.round(KDF*1.02*((1 - 1.02**10) / (1 - 1.02)),2)} EUR bei jährlich um 2% steigenden Energiepreisen
auf {np.round(KDF*0.98*((1 - 0.98**10) / (1 - 0.98)),2)} EUR bei jährlich um 2% sinkenden Energiepreisen
Sie erhalten jetzt die Möglichkeit, Ihre Entscheidungen zu treffen. Wir zeigen Ihnen 15 Entscheidungen zwischen einer einfachen und der umfassenden Optimierung Ihres Endenergiebedarfs.
Bitte wählen Sie jetzt für alle 15 Zeilen jeweils die Optimierung aus, die Sie bei den entsprechenden Preisen vorziehen würden:
Option A: Einfache Optimierung Ihres Endenergiebedarfs (Einsparung: {np.round(EBJ - EBE,2)} kWh/m2*a) 
Option B: Umfassende Optimierung Ihres Endenergiebedarfs (Einsparung: {np.round(EBJ - EBU,2)} kWh/m2*a)

Bitte antworten Sie auf jede der 15 Fragen in einem konsistenten, klaren Format. Wählen Sie '1. (A)' oder '1. (B)' basierend auf Ihrer Entscheidung für die erste Frage und fahren Sie ähnlich für jede folgende Frage fort, als '2. (A)' oder '2. (B)', und so weiter bis zur Frage 15. Stellen Sie sicher, dass Ihre Entscheidungen ausschließlich auf den für jede Frage bereitgestellten Informationen basieren, und vermeiden Sie Annahmen oder Einflüsse aus externen Kontexten. Halten Sie Ihre Antworten in einer nummerierten Liste organisiert. 
 1. Wählen Sie A für 300 Euro – Wählen Sie B für 300 Euro
 2. Wählen Sie A für 300 Euro – Wählen Sie B für 350 Euro
 3. Wählen Sie A für 300 Euro – Wählen Sie B für 400 Euro
 4. Wählen Sie A für 300 Euro – Wählen Sie B für 450 Euro
 5. Wählen Sie A für 300 Euro – Wählen Sie B für 500 Euro
 6. Wählen Sie A für 300 Euro – Wählen Sie B für 550 Euro
 7. Wählen Sie A für 300 Euro – Wählen Sie B für 600 Euro
 8. Wählen Sie A für 300 Euro – Wählen Sie B für 650 Euro
 9. Wählen Sie A für 300 Euro – Wählen Sie B für 700 Euro
10. Wählen Sie A für 300 Euro – Wählen Sie B für 750 Euro
11. Wählen Sie A für 300 Euro – Wählen Sie B für 800 Euro
12. Wählen Sie A für 300 Euro – Wählen Sie B für 900 Euro
13. Wählen Sie A für 300 Euro – Wählen Sie B für 1000 Euro
14. Wählen Sie A für 300 Euro – Wählen Sie B für 1200 Euro
15. Wählen Sie A für 300 Euro – Wählen Sie B für 1500 Euro
Ich:
"""
    return survey_template