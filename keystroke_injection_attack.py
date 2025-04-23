import numpy as np
import pandas as pd
import random

# -----------------------------------------
# Injection and feature recomputation
# -----------------------------------------

def inject_mirror_shift(seq, injection_offset_ms=0, max_len=None):
    """
    Mirror‑shift attack: whenever the user hits Shift (code_norm = 16/255),
    inject a second Shift keystroke with the same press→release latency,
    shifted in time by injection_offset_ms.
    seq: N×7 array [press, release, code_norm, HL, IL, PL, RL]
    injection_offset_ms: ms to delay injected press/release
    max_len: truncate output to this many rows (defaults to original N)
    """
    if max_len is None:
        max_len = seq.shape[0]
    SHIFT = 16/255.0
    # build list of (press, release, code_norm)
    klist = [(r[0], r[1], r[2]) for r in seq]
    new_klist = []
    for press, release, code in klist:
        # always keep the original
        new_klist.append((press, release, code))
        # on any Shift keystroke, mirror it
        if abs(code - SHIFT) < 1e-6:
            off = injection_offset_ms / 1000.0
            hold = release - press
            inj_press   = press + off
            inj_release = inj_press + hold
            new_klist.append((inj_press, inj_release, SHIFT))
    # truncate back to max_len
    new_klist = new_klist[:max_len]

    # recompute HL, IL, PL, RL
    out = np.zeros((len(new_klist), 7))
    for i, (p, r, c) in enumerate(new_klist):
        HL = r - p
        if i < len(new_klist) - 1:
            p2, r2, _ = new_klist[i+1]
            IL = p2 - r
            PL = p2 - p
            RL = r2 - r
        else:
            IL = PL = RL = 0.0
        out[i] = [round(p, 3), round(r, 3), c, round(HL, 3), round(IL, 3), round(PL, 3), round(RL, 3)]
    return out

import numpy as np
import random

def inject_arrows_on_trigger(seq,
                                  down_delay_range_ms=(1, 30),
                                  up_delay_range_ms=(1, 30),
                                  max_len=None):
    """
    Arrow‐injection with human‐like random delays:
      • On each 'e' keystroke (code_norm = 69/255), inject:
          1) Left‐arrow down at e_press + random within down_delay_range_ms
          2) Right‐arrow down at e_press + random within down_delay_range_ms
          3) Original 'e' keyup remains unchanged
          4) Left‐arrow up at e_release + random within up_delay_range_ms
          5) Right‐arrow up at e_release + random within up_delay_range_ms
    seq:                N×7 array [press, release, code_norm, HL, IL, PL, RL]
    down_delay_range_ms: tuple (min, max) ms for arrow‐downs
    up_delay_range_ms:   tuple (min, max) ms for arrow‐ups
    max_len:             truncate to this many rows (default = original N)
    """
    if max_len is None:
        max_len = seq.shape[0]
    TARGET = 69 / 255.0
    LEFT   = 37 / 255.0
    RIGHT  = 39 / 255.0

    # flatten to (press, release, code_norm)
    klist = [(r[0], r[1], r[2]) for r in seq]
    injected = []

    for press, release, code in klist:
        injected.append((press, release, code))
        if abs(code - TARGET) < 1e-6:
            # sample random delays (ms → s)
            ld_off = random.uniform(*down_delay_range_ms) / 1000.0
            rd_off = random.uniform(*down_delay_range_ms) / 1000.0
            lu_off = random.uniform(*up_delay_range_ms)   / 1000.0
            ru_off = random.uniform(*up_delay_range_ms)   / 1000.0

            # schedule injected events
            ld = press   + ld_off
            rd = press   + rd_off
            lu = release + lu_off
            ru = release + ru_off

            injected.append((ld, lu, LEFT))
            injected.append((rd, ru, RIGHT))

    # keep temporal order
    injected.sort(key=lambda x: x[0])
    # print("Injected length:", len(injected))
    injected = injected[:max_len]

    # rebuild feature matrix
    out = np.zeros((len(injected), 7))
    for i, (p, r, c) in enumerate(injected):
        HL = r - p
        if i < len(injected) - 1:
            p2, r2, _ = injected[i+1]
            IL = p2 - r
            PL = p2 - p
            RL = r2 - r
        else:
            IL = PL = RL = 0.0
        out[i] = [round(p, 3), round(r, 3), c, round(HL, 3), round(IL, 3), round(PL, 3), round(RL, 3)]

    return out

# Example usage:
# seq_rand = inject_arrows_on_e_randomized(seq0,
#                                           down_delay_range_ms=(5,20),
#                                           up_delay_range_ms=(80,120))
# print(seq_rand)

import numpy as np
import random

def inject_arrows_random_rate(seq,
                             rate=0.1,
                             down_delay_range_ms=(1, 30),
                             hold_delay_range_ms=(1, 30),
                             max_len=None):
    """
    Randomized arrow‐injection attack:
      • With probability `rate` at each original key,
        inject both Left and Right arrows with human‐like delays.
      • Arrow‐downs occur at original_press + random delay in down_delay_range_ms.
      • Arrow‐ups occur at arrow_down + random hold in hold_delay_range_ms.
    seq:                  N×7 array [press, release, code_norm, HL, IL, PL, RL]
    rate:                 probability of injection per original keystroke
    down_delay_range_ms:  (min, max) delay for arrow‐downs (ms)
    hold_delay_range_ms:  (min, max) hold time for arrow keystrokes (ms)
    max_len:              truncate result to this many rows (default = original N)
    """
    if max_len is None:
        max_len = seq.shape[0]
    LEFT = 37 / 255.0
    RIGHT = 39 / 255.0

    # flatten to (press, release, code_norm)
    klist = [(r[0], r[1], r[2]) for r in seq]
    injected = []

    for press, release, code in klist:
        # keep original event
        injected.append((press, release, code))

        # decide on injection
        if random.random() < rate:
            # sample random offsets
            ld_off = random.uniform(*down_delay_range_ms) / 1000.0
            rd_off = random.uniform(*down_delay_range_ms) / 1000.0
            lh_off = random.uniform(*hold_delay_range_ms) / 1000.0
            rh_off = random.uniform(*hold_delay_range_ms) / 1000.0

            # schedule Left arrow
            ld = press + ld_off
            lu = ld + lh_off
            injected.append((ld, lu, LEFT))

            # schedule Right arrow
            rd = press + rd_off
            ru = rd + rh_off
            injected.append((rd, ru, RIGHT))

    # sort by time and truncate
    injected.sort(key=lambda x: x[0])
    # print("Random RATE =", rate)
    # print("Random RATE Injected length:", len(injected))
    injected = injected[:max_len]

    # rebuild feature matrix
    out = np.zeros((len(injected), 7))
    for i, (p, r, c) in enumerate(injected):
        HL = r - p
        if i < len(injected) - 1:
            p2, r2, _ = injected[i+1]
            IL = p2 - r
            PL = p2 - p
            RL = r2 - r
        else:
            IL = PL = RL = 0.0
        out[i] = [round(p, 3), round(r, 3), c, round(HL, 3), round(IL, 3), round(PL, 3), round(RL, 3)]

    return out

# Example usage:
# seq_rand = inject_arrows_randomized(seq0,
#                                     rate=0.05,
#                                     down_delay_range_ms=(5,20),
#                                     hold_delay_range_ms=(80,120))
# print("Randomized injection shape:", seq_rand.shape)
# print(seq_rand)

def inject_scroll_num_on_trigger(seq,
                                 e_down_delay_range_ms=(1, 30),
                                 e_hold_delay_range_ms=(1, 30),
                                 a_down_delay_range_ms=(1, 30),
                                 a_hold_delay_range_ms=(1, 30),
                                 max_len=None):
    """
    Triggered Scroll/NumLock injection with randomized delays:
      • On 'e' (code 69/255) press:
          - Toggle ScrollLock on at press + random within e_down_delay_range_ms
            and off after random within e_hold_delay_range_ms
            then again toggle off after another down+hold
      • On 'a' (code 65/255) press:
          - Toggle NumLock on at press + random within a_down_delay_range_ms
            and off after random within a_hold_delay_range_ms
            then again toggle off after another down+hold

    seq:      N×7 array [press, release, code_norm, HL, IL, PL, RL]
    e_down_delay_range_ms: (min, max) ms before ScrollLock down on 'e'
    e_hold_delay_range_ms: (min, max) ms between ScrollLock down and up on 'e'
    a_down_delay_range_ms: (min, max) ms before NumLock down on 'a'
    a_hold_delay_range_ms: (min, max) ms between NumLock down and up on 'a'
    max_len:   truncate output to this many rows (default = original N)
    """
    if max_len is None:
        max_len = seq.shape[0]
    # Normalized key codes
    ECODE   = 69  / 255.0
    ACODE   = 65  / 255.0
    SCROLL  = 125 / 255.0
    NUMLOCK = 12  / 255.0

    # Extract original press/release/code
    klist = [(r[0], r[1], r[2]) for r in seq]
    injected = []

    for press, release, code in klist:
        injected.append((press, release, code))
        # On 'e' trigger
        if abs(code - ECODE) < 1e-6:
            # sample delays
            d_off = random.uniform(*e_down_delay_range_ms) / 1000.0
            h_off = random.uniform(*e_hold_delay_range_ms) / 1000.0
            # first toggle on+off
            sd1 = press + d_off
            su1 = sd1 + h_off
            injected.append((sd1, su1, SCROLL))
            # second toggle on+off
            d_off2 = random.uniform(*e_down_delay_range_ms) / 1000.0
            sd2 = su1 + d_off2
            su2 = sd2 + h_off
            injected.append((sd2, su2, SCROLL))
        # On 'a' trigger
        elif abs(code - ACODE) < 1e-6:
            # sample delays
            d_off = random.uniform(*a_down_delay_range_ms) / 1000.0
            h_off = random.uniform(*a_hold_delay_range_ms) / 1000.0
            # first toggle on+off
            nd1 = press + d_off
            nu1 = nd1 + h_off
            injected.append((nd1, nu1, NUMLOCK))
            # second toggle on+off
            d_off2 = random.uniform(*a_down_delay_range_ms) / 1000.0
            nd2 = nu1 + d_off2
            nu2 = nd2 + h_off
            injected.append((nd2, nu2, NUMLOCK))

    # Sort, truncate and rebuild feature matrix
    injected.sort(key=lambda x: x[0])
    injected = injected[:max_len]

    out = np.zeros((len(injected), 7))
    for i, (p, r, c) in enumerate(injected):
        HL = r - p
        if i < len(injected) - 1:
            p2, r2, _ = injected[i+1]
            IL = p2 - r
            PL = p2 - p
            RL = r2 - r
        else:
            IL = PL = RL = 0.0
        out[i] = [round(p, 3), round(r, 3), c,
                  round(HL, 3), round(IL, 3), round(PL, 3), round(RL, 3)]
    return out


def inject_scroll_num_random_rate(seq,
                                  scroll_rate=0.05,
                                  num_rate=0.05,
                                  down_delay_range_ms=(1, 30),
                                  hold_delay_range_ms=(1, 30),
                                  max_len=None):
    """
    Randomized Scroll/NumLock injection with auto-toggle:
      • With probability scroll_rate, toggle ScrollLock twice.
      • With probability num_rate, toggle NumLock twice.
      • Down/off delays sampled from down_delay_range_ms and hold_delay_range_ms.
    """
    if max_len is None:
        max_len = seq.shape[0]
    SCROLL  = 125 / 255.0
    NUMLOCK = 12  / 255.0

    klist = [(r[0], r[1], r[2]) for r in seq]
    injected = [(p, r, c) for p, r, c in klist]

    for press, release, code in klist:
        # ScrollLock toggle
        if random.random() < scroll_rate:
            d1 = random.uniform(*down_delay_range_ms) / 1000.0
            h1 = random.uniform(*hold_delay_range_ms) / 1000.0
            sd1 = press + d1
            su1 = sd1 + h1
            injected.append((sd1, su1, SCROLL))
            # auto-toggle off
            d2 = random.uniform(*down_delay_range_ms) / 1000.0
            sd2 = su1 + d2
            su2 = sd2 + h1
            injected.append((sd2, su2, SCROLL))
        # NumLock toggle
        if random.random() < num_rate:
            d1 = random.uniform(*down_delay_range_ms) / 1000.0
            h1 = random.uniform(*hold_delay_range_ms) / 1000.0
            nd1 = press + d1
            nu1 = nd1 + h1
            injected.append((nd1, nu1, NUMLOCK))
            # auto-toggle off
            d2 = random.uniform(*down_delay_range_ms) / 1000.0
            nd2 = nu1 + d2
            nu2 = nd2 + h1
            injected.append((nd2, nu2, NUMLOCK))

    injected.sort(key=lambda x: x[0])
    # print("Random SCROLL RATE =", scroll_rate)
    # print("Random NUM RATE =", num_rate)
    # print("Injected length:", len(injected))
    injected = injected[:max_len]

    out = np.zeros((len(injected), 7))
    for i, (p, r, c) in enumerate(injected):
        HL = r - p
        if i < len(injected) - 1:
            p2, r2, _ = injected[i+1]
            IL = p2 - r
            PL = p2 - p
            RL = r2 - r
        else:
            IL = PL = RL = 0.0
        out[i] = [round(p, 3), round(r, 3), c, round(HL, 3), round(IL, 3), round(PL, 3), round(RL, 3)]
    return out

# Example usage:
# seq_t = inject_scroll_num_on_trigger(seq0)
# seq_r = inject_scroll_num_random_rate(seq0)


import numpy as np
import random

def inject_backspace_retype(seq,
                            offset_ms_range=(1, 30),
                            d1_ms_range=(1, 30),
                            d2_ms_range=(1, 30),
                            d3_ms_range=(1, 30),
                            max_len=None):
    """
    On each 'e' (code_norm = 69/255) keydown, inject:
      1) Backspace down at press + random offset in offset_ms_range
      2) 'e' retype down at bs_down + random delay in d1_ms_range
      3) Backspace up   at e_retype_down + random delay in d2_ms_range
      4) 'e' retype up  at bs_up + random delay in d3_ms_range
    Original 'e' release remains unchanged.

    seq:      N×7 array [press, release, code_norm, HL, IL, PL, RL]
    offset_ms_range: (min, max) ms before backspace_down
    d1_ms_range:       (min, max) ms between bs_down and e_down
    d2_ms_range:       (min, max) ms between e_down and bs_up
    d3_ms_range:       (min, max) ms between bs_up and e_up
    max_len:           truncate output to this many rows (default = original N)
    """
    # Set default output length
    if max_len is None:
        max_len = seq.shape[0]

    # Constants
    TARGET = 69 / 255.0   # normalized code for 'e'
    BS     = 8  / 255.0   # normalized code for Backspace

    # Extract original events
    klist = [(r[0], r[1], r[2]) for r in seq]
    injected = []

    for press, release, code in klist:
        # Keep original event
        injected.append((press, release, code))

        # Only inject on 'e' keydowns
        if abs(code - TARGET) < 1e-6:
            # Sample randomized delays
            offset_ms = random.uniform(*offset_ms_range)
            d1_ms     = random.uniform(*d1_ms_range)
            d2_ms     = random.uniform(*d2_ms_range)
            d3_ms     = random.uniform(*d3_ms_range)

            # Compute injection times (in seconds)
            bs_down = press + offset_ms / 1000.0
            e_down  = bs_down +    d1_ms   / 1000.0
            bs_up   = e_down  +    d2_ms   / 1000.0
            e_up    = bs_up   +    d3_ms   / 1000.0

            # Inject backspace key press & release
            injected.append((bs_down, bs_up, BS))
            # Inject 'e' retype press & release
            injected.append((e_down,  e_up,  TARGET))

    # Sort all events temporally
    injected.sort(key=lambda x: x[0])
    # print("Injected length:", len(injected))
    # Truncate if needed
    injected = injected[:max_len]

    # Rebuild feature matrix: [press, release, code, HL, IL, PL, RL]
    out = np.zeros((len(injected), 7))
    for i, (p, r, c) in enumerate(injected):
        HL = r - p
        if i < len(injected) - 1:
            p2, r2, _ = injected[i + 1]
            IL = p2 - r
            PL = p2 - p
            RL = r2 - r
        else:
            IL = PL = RL = 0.0
        out[i] = [round(p, 3), round(r, 3), c, round(HL, 3), round(IL, 3), round(PL, 3), round(RL, 3)]

    return out


def inject_bs_retype_random(seq,
                            rate=0.1,
                            bs_delay_range_ms=(1, 30),
                            retype_delay_range_ms=(1, 30),
                            bs_up_delay_range_ms=(1, 30),
                            retype_up_delay_range_ms=(1, 30),
                            max_len=None):
    """
    Randomized backspace+retype injection on letters/numbers:
      • At each keystroke, with probability `rate`, if the key is a letter (A-Z, a-z)
        or digit (0-9), inject:
         1) Backspace down at press + random within bs_delay_range_ms
         2) Re-type the same key down at bs_down + random within retype_delay_range_ms
         3) Backspace up at re_down + random within bs_up_delay_range_ms
         4) Re-type key up at bs_up + random within retype_up_delay_range_ms
      • Original key release remains unchanged.

    seq:                      N×7 array [press, release, code_norm, HL, IL, PL, RL]
    rate:                     injection probability per eligible keystroke
    bs_delay_range_ms:        (min, max) ms before Backspace down
    retype_delay_range_ms:    (min, max) ms between bs_down and retype down
    bs_up_delay_range_ms:     (min, max) ms between retype down and bs_up
    retype_up_delay_range_ms: (min, max) ms between bs_up and retype up
    max_len:                  truncate to this many rows (default = original N)
    """
    if max_len is None:
        max_len = seq.shape[0]
    BS = 8 / 255.0  # backspace
    # Prepare list of original events
    klist = [(r[0], r[1], r[2]) for r in seq]
    injected = []
    for press, release, code in klist:
        injected.append((press, release, code))
        # determine ascii code from normalized
        ascii_code = int(round(code * 255))
        # check if letter or digit
        if ((48 <= ascii_code <= 57) or (65 <= ascii_code <= 90) or (97 <= ascii_code <= 122)) \
           and random.random() < rate:
            # sample delays
            d_bs   = random.uniform(*bs_delay_range_ms) / 1000.0
            d_rt   = random.uniform(*retype_delay_range_ms) / 1000.0
            d_bs_up= random.uniform(*bs_up_delay_range_ms) / 1000.0
            d_rt_up= random.uniform(*retype_up_delay_range_ms) / 1000.0
            # schedule times
            bs_down    = press + d_bs
            re_down    = bs_down + d_rt
            bs_up      = re_down + d_bs_up
            re_up      = bs_up + d_rt_up
            # inject backspace down/up
            injected.append((bs_down, bs_up, BS))
            # inject retype down/up of the same key
            injected.append((re_down, re_up, code))
    # sort and truncate
    injected.sort(key=lambda x: x[0])
    # print("Injected length:", len(injected))
    injected = injected[:max_len]
    # rebuild features
    out = np.zeros((len(injected), 7))
    for i, (p, r, c) in enumerate(injected):
        HL = r - p
        if i < len(injected) - 1:
            p2, r2, _ = injected[i+1]
            IL = p2 - r
            PL = p2 - p
            RL = r2 - r
        else:
            IL = PL = RL = 0.0
        out[i] = [round(p, 3), round(r, 3), c, round(HL, 3), round(IL, 3), round(PL, 3), round(RL, 3)]
    return out

# Example usage:
# seq_rand = inject_bs_retype_random(seq0,
#                                    rate=0.1,
#                                    bs_delay_range_ms=(5,20),
#                                    retype_delay_range_ms=(20,50),
#                                    bs_up_delay_range_ms=(20,50),
#                                    retype_up_delay_range_ms=(20,50))
# print("Random backspace+retype injection shape:", seq_rand.shape)
# print(seq_rand)








# -----------------------------------------
# Data Loading and Sequence Extraction
# -----------------------------------------
input_file = "./dataset_dont_commit/participants_subset_68001_to_69000_normalized_padded.csv"
df = pd.read_csv(input_file)

# Normalize participant IDs to [0...]
unique_parts = df['PARTICIPANT_ID'].drop_duplicates().to_numpy()
norm_map = {pid: idx for idx, pid in enumerate(unique_parts)}
users_to_use = list(norm_map.values())[:1000]

# Build per-user sequence dict
user_sequences_dict = {u: [] for u in users_to_use}
for pid in unique_parts[:1000]:
    u = norm_map[pid]
    sub = df[df['PARTICIPANT_ID'] == pid]
    for sid in sub['TEST_SECTION_ID'].drop_duplicates():
        arr = sub[sub['TEST_SECTION_ID'] == sid].to_numpy()[:, -7:]
        user_sequences_dict[u].append(arr)
for u in user_sequences_dict:
    user_sequences_dict[u] = np.array(user_sequences_dict[u])
    
    
    
    

# np.set_printoptions(linewidth=200, formatter={'all':lambda x: str(x)})
# # -----------------------------------------
# # Example Usage
# # -----------------------------------------
# # Select first user, first sequence
# seq0 = user_sequences_dict[0][0]
# print("Original sequence shape:", seq0.shape)
# print("Original sequence:\n", seq0)
# print()

# # # Method 1: inject on 'e' with 50ms delay, 10ms offset
# # seq1 = inject_arrows_on_e_old(seq0, human_delay_ms=50, injection_offset_ms=10)
# # print("After on-e injection shape:", seq1.shape)
# # print("After on-e injection:\n", seq1)
# # print()

# # # Method 2: random injection at 5% rate
# # seq2 = inject_arrows_random_old(seq0, rate=0.05, human_delay_ms=50)
# # print("After random injection shape:", seq2.shape)
# # print("After random injection:\n", seq2)
# # print()

# seq3 = inject_arrows_on_trigger(seq0)
# print("ARROWS ON TRIGGER shape:", seq3.shape)
# print("ARROWS ON TRIGGER injection:\n", seq3)
# print()

# seq4 = inject_arrows_random_rate(seq0)
# print("ARROWS RANDOM RATE shape:", seq4.shape)
# print("ARROWS RANDOM RATE injection:\n", seq4)
# print()

# seq4 = inject_scroll_num_on_trigger(seq0)
# print("SCROLL & NUM LOCK ON TRIGGER shape:", seq4.shape)
# print("SCROLL & NUM LOCK ON TRIGGER injection:\n", seq4)
# print()

# seq5 = inject_scroll_num_random_rate(seq0)
# print("After randomized injection shape:", seq5.shape)
# print("After randomized injection:\n", seq5)
# print()

# seq6 = inject_backspace_retype(seq0)
# print("After backspace retype injection shape:", seq6.shape)
# print("After backspace retype injection:\n", seq6)
# print()

# seq7 = inject_bs_retype_random(seq0)
# print("After randomized backspace retype injection shape:", seq7.shape)
# print("After randomized backspace retype injection:\n", seq7)
# print()

# # -----------------------------------------