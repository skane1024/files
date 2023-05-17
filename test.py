from Crypto.Cipher import AES   #pip install pycryptodome
import os

def pad(data):
    pad_len = AES.block_size - len(data) % AES.block_size
    return data + pad_len * bytes([pad_len])

def unpad(data):
    pad_len = data[-1]
    return data[:-pad_len]

def encrypt_file(key, input_file, output_file):
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv
    with open(input_file, 'rb') as fin:
        with open(output_file, 'wb') as fout:
            fout.write(iv)
            while True:
                chunk = fin.read(1024 * AES.block_size)
                if not chunk:
                    break
                if len(chunk) % AES.block_size != 0:
                    chunk = pad(chunk)
                fout.write(cipher.encrypt(chunk))

def decrypt_file(key, input_file, output_file):
    with open(input_file, 'rb') as fin:
        with open(output_file, 'wb') as fout:
            iv = fin.read(AES.block_size)
            cipher = AES.new(key, AES.MODE_CBC, iv=iv)
            while True:
                chunk = fin.read(1024 * AES.block_size)
                if not chunk:
                    break
                data = cipher.decrypt(chunk)
                if len(data) % AES.block_size != 0:
                    data = unpad(data)
                fout.write(data)

# 使用 16 字节的 key 加密文件
key = os.urandom(16)
print(key)

# # 加密文件
encrypt_file(key, 'myfile.bin', 'myfile_encrypted.bin')

# 解密文件
decrypt_file(key, 'myfile_encrypted.bin', 'myfile_decrypted.bin')




import os

# 定义要拆分的文件和每个文件的大小（单位为字节）
input_file = 'myfile.bin'
chunk_size = 1024 * 1024  # 每个文件大小为 1MB

# 打开输入文件，读取数据并拆分为多个块
with open(input_file, 'rb') as fin:
    file_index = 0
    while True:
        chunk = fin.read(chunk_size)
        if not chunk:
            break
        # 如果块大小不足 chunk_size，就用 0 补齐
        if len(chunk) < chunk_size:
            chunk += bytes(chunk_size - len(chunk))
        # 写入新的文件
        output_file = f'myfile_{file_index:03d}.bin'
        with open(output_file, 'wb') as fout:
            fout.write(chunk)
        file_index += 1

print(f'Split {input_file} into {file_index} files.')

