import os
import pandas as pd
import vectorize

def labeling(folder, output_csv='vectors\\data.csv'):
    all_data = []

    # ✅ Extract label from folder name
    label = os.path.basename(folder).split('_')[1]

    # ✅ Load existing CSV safely
    if os.path.exists(output_csv) and os.path.getsize(output_csv) > 0:
        existing_df = pd.read_csv(output_csv)
    else:
        existing_df = pd.DataFrame(columns=vectorize.columns)

    # ✅ Continue ID correctly
    video_id = int(existing_df["id"].max()) + 1 if not existing_df.empty else 1

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)

        if os.path.isfile(file_path) and filename.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"Processing: {filename}")

            try:
                df = vectorize.vetorize(
                    file_path,
                    video_id=video_id,
                    label=label
                )

                if not df.empty:
                    all_data.append(df)
                    video_id += 1
                else:
                    print(f"⚠️ Empty result: {filename}")

            except Exception as e:
                print(f"❌ Error with {filename}: {e}")

    if all_data:
        new_data = pd.concat(all_data, ignore_index=True)
        final_df = pd.concat([existing_df, new_data], ignore_index=True)

        final_df.to_csv(output_csv, index=False)
        print(f"✅ Saved {len(new_data)} rows to CSV")

    else:
        print("⚠️ No valid data extracted")


# ▶ Run
if __name__ == "__main__":
    labeling('videos\\barev_0')