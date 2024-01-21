from classes.Node import Node
from PrettyPrint import PrettyPrintTree


class Drawer:
    def __init__(self, root: Node):
        self.xStep = 5
        self.yStep = 5
        self.maxX = 0
        self.minY = 0
        self.texts = []
        self.root = root

    def update(self, node: Node):
        # pt = PrettyPrintTree(
        #     lambda x: [y for y in x.children if y.visitCount > 0],
        #     lambda x: x.val,
        #     max_depth=-1,
        #     return_instead_of_print=True,
        #     color=None,
        # )
        pt = PrettyPrintTree(
            lambda x: [y for y in x.children],  # type: ignore
            lambda x: x.val,  # type: ignore
            max_depth=2,
            return_instead_of_print=True,
            color=None,  # type: ignore
        )
        tree_as_str = pt(node)
        # with open('tree.txt', 'w', encoding="utf8") as f:
        #     f.write(tree_as_str)
        from PIL import Image
        from PIL import ImageDraw
        from PIL import ImageFont

        img = Image.new("RGB", (100, 100))
        d = ImageDraw.Draw(img)
        fontSize = 20
        font = ImageFont.truetype("FreeMono.ttf", size=fontSize)

        _, _, width, height = d.textbbox(text=tree_as_str, font=font, xy=(0, 0))  # type: ignore
        img = Image.new("RGB", (width + 2 * fontSize, height + 2 * fontSize))
        d = ImageDraw.Draw(img)
        d.text((20, 20), tree_as_str, fill=(255, 255, 255), font=font)  # type: ignore
        img.show()
        img.save("tree.tiff")
        # exit()
