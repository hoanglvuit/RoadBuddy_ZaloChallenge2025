from huggingface_hub import create_repo, upload_folder

# ====== CONFIG ======
repo_id = "hoanglvuit/qwen2.5vl"   # sửa tên repo bạn muốn
folder_path = "unsloth_finetune"  # đường dẫn tới folder chứa model
# ====================

print(f"🔧 Creating repo (public): {repo_id} ...")
create_repo(repo_id, private=False, exist_ok=True)

print(f"🚀 Uploading folder: {folder_path} ...")
upload_folder(
    folder_path=folder_path,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload Qwen-VL model"
)

print("🎉 DONE! Model has been uploaded to HuggingFace.")
print(f"➡️  Link: https://huggingface.co/{repo_id}")
