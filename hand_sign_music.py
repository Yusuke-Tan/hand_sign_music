import cv2
import mediapipe as mp
import pygame
import numpy as np
import time

# --- 1. 設定と準備 ---
pygame.init()
pygame.mixer.set_num_channels(8)
pygame.mixer.init(frequency=44100, size=-16, channels=2)
SAMPLE_RATE = 44100

# バスドラム（Kick）音生成関数
def create_kick_sound(duration=0.4):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    freqs = 40 + 100 * np.exp(-40 * t)
    phase = 2 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
    wave = np.sin(phase) * np.exp(-12 * t)
    wave = wave / np.max(np.abs(wave)) * 0.9
    wave = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((wave, wave)))

# スネアドラム音生成関数
def create_snare_sound(duration=0.4):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    freqs = 180 + 50 * np.exp(-20 * t)
    phase = 2 * np.pi * np.cumsum(freqs) / SAMPLE_RATE
    tone = np.sin(phase) * np.exp(-15 * t)
    noise = np.random.uniform(-1.0, 1.0, len(t)) * np.exp(-15 * t)
    wave = (tone * 0.5 + noise * 0.5)
    wave = wave / np.max(np.abs(wave)) * 0.7
    wave = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((wave, wave)))

# ハイハット音生成関数
def create_hihat_sound(duration=0.15):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    noise = np.random.uniform(-1.0, 1.0, len(t))
    envelope = np.exp(-40 * t)
    wave = noise * envelope * 0.25
    wave = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((wave, wave)))

# ベース音生成関数 (少し倍音を加えてベースらしさを出します)
def create_bass_sound(frequency, duration=0.3):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    wave = 0.7 * np.sin(2 * np.pi * frequency * t)
    wave += 0.3 * np.sin(2 * np.pi * frequency * 2 * t)
    envelope = np.exp(-5 * t)
    wave = wave * envelope
    wave = wave / np.max(np.abs(wave)) * 0.8
    wave = (wave * 32767).astype(np.int16)
    return pygame.sndarray.make_sound(np.column_stack((wave, wave)))

# 楽器の音を辞書に登録
sounds = {
    "Kick": create_kick_sound(),
    "Snare": create_snare_sound(),
    "HiHat": create_hihat_sound(),
    # ベースの音階を用意 (C, F, G)
    "Bass_C": create_bass_sound(65.41), # C2
    "Bass_F": create_bass_sound(87.31), # F2
    "Bass_G": create_bass_sound(98.00)  # G2
}

# 指の状態判定
def get_finger_status(landmarks):
    fingers = []
    if landmarks[4].x < landmarks[3].x: fingers.append(True)
    else: fingers.append(False)
    for tip, pip in zip([8, 12, 16, 20], [6, 10, 14, 18]):
        if landmarks[tip].y < landmarks[pip].y: fingers.append(True)
        else: fingers.append(False)
    return fingers

# --- メイン処理 ---
def main():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
    mp_drawing = mp.solutions.drawing_utils
    cap = cv2.VideoCapture(0)
    
    # 状態管理用の変数
    playing_parts = set()  # 現在演奏しているパート
    pending_parts = set()  # 次のループから演奏を開始するパート（同期用）
    
    current_step = 0          
    bpm = 100                 
    step_interval = 60 / (bpm * 2) 
    last_play_time = 0
    
    rock_start_time = None    
    wrist_x_history = []      

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        display_text = ""
        color = (255, 255, 255)

        if results.multi_hand_landmarks and results.multi_handedness:
            hand_landmarks = results.multi_hand_landmarks[0]
            handedness = results.multi_handedness[0].classification[0].label
            
            if handedness == "Right":
                f = get_finger_status(hand_landmarks.landmark)
                total_fingers = sum(f)
                wrist_x = hand_landmarks.landmark[0].x

                # --- 1. パー（✋）にして左右に振る判定（ドラムパート追加） ---
                if total_fingers == 5:
                    rock_start_time = None 
                    wrist_x_history.append(wrist_x)
                    if len(wrist_x_history) > 15:
                        wrist_x_history.pop(0)
                        if max(wrist_x_history) - min(wrist_x_history) > 0.15:
                            # まだ何も鳴っていない場合は即座に開始
                            if not playing_parts and not pending_parts:
                                playing_parts.add("Drums")
                                current_step = 0
                                last_play_time = time.time() - step_interval
                            # 既に他の楽器が鳴っている場合は待機リストに追加
                            elif "Drums" not in playing_parts:
                                pending_parts.add("Drums")
                            wrist_x_history.clear()
                            
                # --- 2. チョキ（✌）にして左右に振る判定（ベースパート追加） ---
                elif total_fingers == 2 and f[1] and f[2]:
                    rock_start_time = None 
                    wrist_x_history.append(wrist_x)
                    if len(wrist_x_history) > 15:
                        wrist_x_history.pop(0)
                        if max(wrist_x_history) - min(wrist_x_history) > 0.15:
                            if not playing_parts and not pending_parts:
                                playing_parts.add("Bass")
                                current_step = 0
                                last_play_time = time.time() - step_interval
                            elif "Bass" not in playing_parts:
                                pending_parts.add("Bass")
                            wrist_x_history.clear()
                
                # --- 3. グー（✊）を1秒握る判定（すべてストップ） ---
                elif total_fingers == 0:
                    wrist_x_history.clear() 
                    
                    if rock_start_time is None:
                        rock_start_time = time.time()
                    else:
                        elapsed_time = time.time() - rock_start_time
                        if elapsed_time >= 0.5:
                            playing_parts.clear()
                            pending_parts.clear()
                            current_step = 0
                            display_text = "STOPPED"
                            color = (0, 0, 255)
                        else:
                            display_text = "Stopping..."
                            color = (0, 165, 255)
                
                else:
                    rock_start_time = None
                    wrist_x_history.clear()

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
        else:
            rock_start_time = None
            wrist_x_history.clear()

        # --- シーケンサー処理（リズム同期） ---
        if playing_parts:
            current_time = time.time()
            if current_time - last_play_time >= step_interval:
                
                # ★ポイント: ステップが「0」（ループの頭）になった瞬間に、待機中の楽器を合流させる
                if current_step == 0 and pending_parts:
                    playing_parts.update(pending_parts)
                    pending_parts.clear()

                log_messages = [f"[{current_step + 1}/8]"]

                # ドラムの再生制御
                if "Drums" in playing_parts:
                    sounds["HiHat"].play()
                    if current_step == 0 or current_step == 4:
                        sounds["Kick"].play()
                        log_messages.append("Drums: Kick+Hat")
                    elif current_step == 2 or current_step == 6:
                        sounds["Snare"].play()
                        log_messages.append("Drums: Snare+Hat")
                    else:
                        log_messages.append("Drums: Hat")

                # ベースの再生制御（簡単なコード進行 C -> F -> G）
                if "Bass" in playing_parts:
                    if current_step in [0, 1, 2, 3]:
                        sounds["Bass_C"].play()
                        log_messages.append("Bass: C")
                    elif current_step in [4, 5]:
                        sounds["Bass_F"].play()
                        log_messages.append("Bass: F")
                    elif current_step in [6, 7]:
                        sounds["Bass_G"].play()
                        log_messages.append("Bass: G")

                # コンソールに現在の演奏状況を出力
                print(" | ".join(log_messages))
                
                # 次のステップへ
                current_step = (current_step + 1) % 8
                last_play_time = current_time
            
            # 画面表示テキストの更新
            if display_text == "":
                playing_text = " + ".join(playing_parts)
                if pending_parts:
                    display_text = f"Playing: {playing_text} (Next: {', '.join(pending_parts)})"
                    color = (0, 255, 255)
                else:
                    display_text = f"Playing: {playing_text}"
                    color = (0, 255, 100)

        if display_text:
            cv2.putText(frame, display_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        cv2.imshow('Hand Rhythm Machine', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()

if __name__ == "__main__":
    main()