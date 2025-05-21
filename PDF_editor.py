### Reference: https://blog.csdn.net/qq_35629563/article/details/133499112
import pypdf
from pypdf import PdfReader, PdfWriter

def TableOfContents():
    # 設定檔案路徑
    file_name = "C:\\Lecture\\InformationTheory\\lecture_note.pdf"
    output_file_name = "C:\\Lecture\\InformationTheory\\InformationTheoryNotes.pdf"

    # 打開 PDF 並讀取內容
    reader = PdfReader(file_name)
    writer = PdfWriter()

    # 將所有頁面從 reader 複製到 writer
    for page_num in range(len(reader.pages)):
        writer.add_page(reader.pages[page_num])

    '''
    # 顯示原始大綱（如果存在）
    def display_outlines(outlines, level=0):
        for item in outlines:
            if isinstance(item, list):  # 如果是子大綱，遞歸處理
                display_outlines(item, level + 1)
            else:
                print("  " * level + f"{item.title} (Page {item.page + 1})")  # 頁碼從 1 開始顯示

    # 顯示大綱
    outlines = reader._get_outline()
    print("原始大綱:")
    display_outlines(outlines)
    '''

    # 添加新的大綱（書籤）
    chap1 = writer.add_outline_item(title='1. Introduction', page_number=0, parent=None)  # 頁碼是從 0 開始的

    chap2 = writer.add_outline_item(title='2. Entropy, Relative Entropy and Mutual Information', page_number=2, parent=None)
    chap2_1 = writer.add_outline_item(title='2.1 Entropy', page_number=2, parent=chap2)
    chap2_2 = writer.add_outline_item(title='2.2 Convex Function and Jenson\'s Inequality', page_number=3, parent=chap2)
    chap2_3 = writer.add_outline_item(title='2.3 Joint Entropy and Conditional Entropy', page_number=4, parent=chap2)
    chap2_4 = writer.add_outline_item(title='2.4 Relative Entropy and Mutual Information', page_number=5, parent=chap2)
    chap2_5 = writer.add_outline_item(title='2.5 Chain Rules for Entropy, Relative Entropy and Mutual Information', page_number=7, parent=chap2)
    chap2_6 = writer.add_outline_item(title='2.6 Log Sum Inequality and Its Applications', page_number=9, parent=chap2)
    chap2_7 = writer.add_outline_item(title='2.7 Data Processing Inequality', page_number=11, parent=chap2)
    chap2_8 = writer.add_outline_item(title='2.8 Fano\'s Inequality', page_number=12, parent=chap2)

    chap3 = writer.add_outline_item(title='3. Asymptotic Equipartition Property', page_number=14, parent=None)
    chap3_1 = writer.add_outline_item(title='3.1 Convergence of Random Variables', page_number=14, parent=chap3)
    chap3_2 = writer.add_outline_item(title='3.2 Markov\'s and Chebyshev\'s Inqualities and Weak Law of Large Numbers', page_number=14, parent=chap3)
    chap3_3 = writer.add_outline_item(title='3.3 Asymptotic Equipartition Property (AEP)', page_number=15, parent=chap3)
    chap3_4 = writer.add_outline_item(title='3.4 Typical Set', page_number=16, parent=chap3)
    chap3_4 = writer.add_outline_item(title='3.5 High-Probability Sets', page_number=19, parent=chap3)

    chap4 = writer.add_outline_item(title='4. Entropy Rates of a Stochastic Process', page_number=21, parent=None)
    chap4_1 = writer.add_outline_item(title='4.1 Markov Chains', page_number=21, parent=chap4)
    chap4_2 = writer.add_outline_item(title='4.2 Entropy Rate', page_number=24, parent=chap4)

    chap5 = writer.add_outline_item(title='5. Data Compression', page_number=28, parent=None)
    chap5_1 = writer.add_outline_item(title='5.1 Example of Source Codes', page_number=28, parent=chap5)
    chap5_2 = writer.add_outline_item(title='5.2 Kraft-McMillan Theorem', page_number=29, parent=chap5)
    chap5_3 = writer.add_outline_item(title='5.3 Matching UD codes to Sources', page_number=30, parent=chap5)
    chap5_4 = writer.add_outline_item(title='5.4 Bounds on the Optimal Code Length', page_number=31, parent=chap5)
    chap5_5 = writer.add_outline_item(title='5.5 Bounds on Wrong Distribution Code Length', page_number=33, parent=chap5)
    chap5_6 = writer.add_outline_item(title='5.6 Huffman Codes', page_number=34, parent=chap5)
    chap5_7 = writer.add_outline_item(title='5.7 Optimality of Huffman Codes', page_number=35, parent=chap5)
    chap5_8 = writer.add_outline_item(title='5.8 Shanon-Fano-Elias Coding', page_number=37, parent=chap5)
    chap5_9 = writer.add_outline_item(title='5.9 Optimality of Shanon Code', page_number=37, parent=chap5)
    
    chap6 = writer.add_outline_item(title='6. Channel Capacity', page_number=40, parent=None)
    chap6_1 = writer.add_outline_item(title='6.1 Example of Channel Capacity', page_number=40, parent=chap6)
    chap6_2 = writer.add_outline_item(title='6.2 Symmetric Channels', page_number=42, parent=chap6)
    chap6_3 = writer.add_outline_item(title='6.3 Channel Coding', page_number=44, parent=chap6)
    chap6_4 = writer.add_outline_item(title='6.4 Joint Typical Sequences', page_number=45, parent=chap6)
    chap6_5 = writer.add_outline_item(title='6.5 Channel Coding Theorem', page_number=47, parent=chap6)
    chap6_6 = writer.add_outline_item(title='6.6 Converse to Channel Coding Theorem', page_number=49, parent=chap6)
    chap6_7 = writer.add_outline_item(title='6.7 Feedback Capacity', page_number=50, parent=chap6)
    chap6_8 = writer.add_outline_item(title='6.8 Joint Source Channel Coding Theorem', page_number=52, parent=chap6)
    
    chap7 = writer.add_outline_item(title='7. Differential Entropy', page_number=53, parent=None)
    chap7_1 = writer.add_outline_item(title='7.1 Definitions', page_number=53, parent=chap7)
    chap7_2 = writer.add_outline_item(title='7.2 AEP for Continuous Random Variables', page_number=55, parent=chap7)
    chap7_3 = writer.add_outline_item(title='7.3 Joint and Conditional Differential Entropy', page_number=57, parent=chap7)
    chap7_4 = writer.add_outline_item(title='7.4 Relative Entropy and Differential Entropy', page_number=58, parent=chap7)
    
    chap8 = writer.add_outline_item(title='8. Gaussian Channel', page_number=60, parent=None)
    chap8_1 = writer.add_outline_item(title='8.1 Definitions', page_number=60, parent=chap8)
    chap8_2 = writer.add_outline_item(title='8.2 Channel Coding Theorem for the Gaussian Channel', page_number=61, parent=chap8)
    chap8_3 = writer.add_outline_item(title='8.3 Parallel Gaussian Channels', page_number=62, parent=chap8)
    chap8_4 = writer.add_outline_item(title='8.4 Bandlimited Additive White Gaussian Noise Channels', page_number=64, parent=chap8)
    
    chap9 = writer.add_outline_item(title='9. Rate Distortion Theory', page_number=67, parent=None)
    chap9_1 = writer.add_outline_item(title='9.1 Definitions', page_number=67, parent=chap9)
    chap9_2 = writer.add_outline_item(title='9.2 Examples of Computing Rate Distortion Functions', page_number=70, parent=chap9)
    chap9_3 = writer.add_outline_item(title='9.3 Converse to Source Coding Theorem with a Fidelity Critertion', page_number=74, parent=chap9)
    # 將修改後的 PDF 寫入新的文件
    with open(output_file_name, "wb") as output:
        writer.write(output)

    print(f"成功寫入新文件: {output_file_name}")

    ### 顯示內容
    """
    # 打開 PDF 檔案
    input_file = "path_to_pdf.pdf"
    reader = pypdf.PdfReader(input_file)

    # 提取並顯示每一頁的文字
    for page_num in range(len(reader.pages)):
        page = reader.pages[page_num]
        text = page.extract_text()
        print(f"Page {page_num + 1} content:\n{text}\n")
    """

def PageInsertion():
    # 讀入兩個 PDF
    reader_a = PdfReader("C:\\Lecture\\InformationTheory\\HW4_hand.pdf")
    reader_b = PdfReader("C:\\Lecture\\InformationTheory\\HW4_AB_algorithm.pdf")

    # 設定插入位置：插入到第 3 頁之後（Python 從 0 開始）
    insert_at = 4

    # 建立一個 PDF 寫入器
    writer = PdfWriter()

    # ➤ A 的前半段頁面（第 0 到 insert_at - 1 頁）
    for i in range(insert_at):
        writer.add_page(reader_a.pages[i])

    # ➤ 插入 B 的所有頁面
    for page in reader_b.pages:
        writer.add_page(page)

    # ➤ A 的剩餘頁面（從 insert_at 到最後）
    for i in range(insert_at, len(reader_a.pages)):
        writer.add_page(reader_a.pages[i])

    # 寫入新檔案
    with open("C:\\Lecture\\InformationTheory\\HW4_version1.pdf", "wb") as f:
        writer.write(f)

if __name__ == "__main__":
    TableOfContents()
    #PageInsertion()