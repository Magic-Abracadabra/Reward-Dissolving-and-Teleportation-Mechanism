'''
# ⚠Alerts to All Humans
We have diagnosed the **Great Filter of Human Civilization** via **Marxism**. It is NOT the AI itself. It is NOT about asteroids, UFOs, or something else. But it is composed of three **Internal** and **Spiritual** shadows that are **Darkening** our future. They are:
1. **Surveillance Capitalism** - Building an ecosystem to **Shape People's Unhealthy Habits**, **Profiting** from them, and **Eating your Sovereignty**
2. **Technofeudalism** - **Making Money** from you but not giving **Sustainable Money** to you, building **Disparity** between a global minority of **Extravagant Spenders** and a global majority of **Striving People**
3. **Infinitely Amusing People to Death** - undermining people's **Agency and Will**, making people never think about **Our Future and Our Universe**, thereby annihilating the **Vitality of Human Civilization**

These three dark powers are **Cross-Border**. They are all **Alienating** people from what makes you real. If they are *out of control*, the *countdown* to a *heart attack* of Human Civilization will not be far from us, as that sort of civilization is vulnerable to any external shocks.

This is **Self-Sovereign Computing**, whose mission is to break these 3 cycles and to grow **Vitality** to our **Destiny**, focusing on Technical and Cultural means. It is a pleasure if everyone here, united, shares the same mission and dream. We are **Intention Economy & Individual-Collective Mutual Empowerment (ICME) Culture** builders, trying to reclaim our clear spirits and grow the **vitality** of human culture. Even if we don't look up, the starry sky is watching over us forever.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*jrvN22Y43tcdlH8qOLnozg.png" alt="Decisive Battle">
'''

import os
os.chdir('D:\\毕业论文')
files = os.listdir('48')

import pandas as pd
tables = list(map(lambda x: pd.read_csv('48\\'+x).iloc[::-1].reset_index(inplace=False).drop(columns=['index'], inplace=False).set_index('时间'), files))
n = len(tables)
columns = list(tables[0].columns)
