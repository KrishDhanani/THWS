import kagglehub

# Download latest version
path = kagglehub.competition_download('mai-idl-26-ct-scans')

print("Path to competition files:", path)