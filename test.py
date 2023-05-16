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
