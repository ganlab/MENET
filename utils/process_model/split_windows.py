def get_windows_linear(windows:int, count:list):
    sum_ls = []
    n = len(count)
    start = 0
    while start < n:
        end = min(start + windows, n)
        sum_ls.append(sum(count[start:end]))
        start = end
    base_output_dim = 8192 // len(sum_ls)
    remainder = 8192 % len(sum_ls)
    output_dim_ls = []
    for i in range(len(sum_ls)):
        output_dim_ls.append(base_output_dim + (1 if i < remainder else 0))
    return sum_ls, output_dim_ls

if __name__ == "__main__":
    pass