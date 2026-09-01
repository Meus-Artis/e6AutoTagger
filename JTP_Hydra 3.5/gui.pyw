import argparse
import csv
import difflib
import os
import platform
import queue
import random
import re
import sys
import time
from collections import Counter, defaultdict, deque
from io import StringIO
from itertools import chain, product
from math import ceil, exp, log
from traceback import print_exc
from threading import Thread, Event, Lock
from typing import Any, Callable, Iterable, Sequence, TypeAlias, TypeVar, ParamSpec, cast
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"
import torch
from torch import Tensor
from einops import rearrange
from PySide6.QtCore import (
    Qt, QObject, QMetaObject, Signal, SignalInstance, Slot, Property, QEvent,
    QPoint, QMargins, QRect, QSize, QUrl, QDir, QChildEvent,
    QKeyCombination, QMimeData, QEventLoop,
)
from PySide6.QtGui import (
    QGuiApplication, QIcon, QKeySequence, QWindow,
    QFont, QTextCursor, QTextOption, QPalette, QColor, QColorSpace,
    QDragEnterEvent, QDropEvent, QImage, QPixmap, QResizeEvent,
    QContextMenuEvent, QAction, QMouseEvent, QKeyEvent,
)
from PySide6.QtWidgets import *
import requests
from hydra import glu
from hydra.siglip2 import HAS_VARLEN_ATTN, NaFlexFeatures, NaFlexFeaturesVarlen
from hydra.label import Label, Rewriter, TAG_CATEGORIES, parse_list, parse_expandable_list, parse_aliases
from hydra.classification import (
    IMPLICATION_MODES, Calibration, ExclusiveGroup,
    f_score, csi, min_precision, log_interval, interval_exp
)
from hydra.model import Hydra, Extension, load_model
from hydra import image
from hydra.image import Image
from inference import process_batched, paths_iter
from utils.loader import Loader
from utils.workqueue import WorkQueue
from huggingface_hub import hf_hub_download
import shutil
W = TypeVar("W", bound=QWidget)
L = TypeVar("L", bound=QLayout)
P = ParamSpec("P")
CATEGORY_COLORS = {
    "artist": QColor.fromString("#f2ac08"),
    "contributor": QColor.fromString("#f2ac08"),
    "copyright": QColor.fromString("#dd00dd"),
    "character": QColor.fromString("#00aa00"),
    "species": QColor.fromString("#ed5d1f"),
    "meta": QColor.fromString("#2e76b4"),
    "invalid": QColor.fromString("#ff3d3d"),
    "lore": QColor.fromString("#228822"),
}
print("Checking for model...")
if not os.path.isfile("JTP-Hydra-3.5.safetensors"):
    print("Downloading tagger model, please wait...")
    path = hf_hub_download(repo_id="RedRocket/Hydra", filename="models/hydra-3.5.safetensors", repo_type="model", local_dir="./")
    shutil.move(path, "JTP-Hydra-3.5.safetensors")
    os.rmdir("models")
else:
    print("JTP-Hydra-3.5.safetensors found")

def connect(*signals: SignalInstance) -> Callable[[Callable[P, None]], Callable[P, None]]:
    def _connect(fn: Callable[P, None]) -> Callable[P, None]:
        for signal in signals:
            signal.connect(fn)

        return fn

    return _connect

class MainWindow(QMainWindow):
    modifiersChanged = Signal(Qt.KeyboardModifier)

    def __init__(self) -> None:
        super().__init__()

        self.keyboardModifiers = Qt.KeyboardModifier.NoModifier

        self.setWindowTitle("RedRocket Hydra Classifier")

        try:
            self.setWindowIcon(QIcon("data/icon.png"))
        except:
            pass

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)

        if event.type() == QEvent.Type.ActivationChange:
            self.checkModifiers()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        super().keyPressEvent(event)
        self._checkModifiers(event.modifiers())

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        super().keyReleaseEvent(event)
        self._checkModifiers(event.modifiers())

    def _checkModifiers(self, modifiers: Qt.KeyboardModifier) -> None:
        if modifiers != self.keyboardModifiers:
            self.keyboardModifiers = modifiers
            self.modifiersChanged.emit(modifiers)

    def checkModifiers(self) -> None:
        self._checkModifiers(
            QGuiApplication.queryKeyboardModifiers()
            if self.isActiveWindow()
            else Qt.KeyboardModifier.NoModifier
        )

MAIN_WINDOW: MainWindow

class ContextualAction(QObject):
    changed = Signal()
    triggered = Signal()

    def __init__(
        self,
        text: str = "",
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        shortcut: QKeySequence | None = None,
        tooltip: str = "",
        icon: QIcon | None = None,
        parent: "ContextualButton | None" = None,
    ) -> None:
        super().__init__()

        if not tooltip:
            clean_text = text.replace("&", "").replace("…", "")
            tooltip = f"[%s] {clean_text}"

        self._modifiers = modifiers
        self._shortcut = shortcut if shortcut is not None else QKeySequence()
        self._text = text
        self._icon = icon if icon is not None else QIcon()
        self._tooltip = tooltip
        self._visible = True
        self._enabled = True

        if parent is not None:
            self.setParent(parent)

    def modifiers(self) -> Qt.KeyboardModifier:
        return self._modifiers

    def setModifiers(self, value: Qt.KeyboardModifier | None) -> None:
        self._modifiers = value if value is not None else Qt.KeyboardModifier.NoModifier
        self.changed.emit()

    def shortcut(self) -> QKeySequence:
        return self._shortcut

    def setShortcut(self, value: QKeySequence | None) -> None:
        self._shortcut = value if value is not None else QKeySequence()
        self.changed.emit()

    def text(self) -> str:
        return self._text

    def setText(self, value: str) -> None:
        self._text = value
        self.changed.emit()

    def icon(self) -> QIcon:
        return self._icon

    def setIcon(self, value: QIcon | None) -> None:
        self._icon = value if value is not None else QIcon()
        self.changed.emit()

    def toolTip(self) -> str:
        return self._tooltip

    def setToolTip(self, value: str) -> None:
        self._tooltip = value
        self.changed.emit()

    def visible(self) -> bool:
        return self._visible

    def setVisible(self, value: bool = True) -> None:
        self._visible = value
        self.changed.emit()

    def enabled(self) -> bool:
        return self._enabled

    def setEnabled(self, value: bool = True) -> None:
        self._enabled = value
        self.changed.emit()

    def query(self, modifiers: Qt.KeyboardModifier) -> bool:
        return self._visible and self._modifiers in modifiers

    def getDisplayToolTip(self) -> str:
        return self.toolTip().replace(
            "%s",
            self.shortcut().toString(QKeySequence.SequenceFormat.NativeText)
        )

    def trigger(self) -> None:
        self.triggered.emit()

class ContextualButton(QPushButton):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._enabled = True
        self._frozen = False

        MAIN_WINDOW.modifiersChanged.connect(self.contextChanged)
        self.clicked.connect(self._clicked)

    def childEvent(self, event: QChildEvent) -> None:
        super().childEvent(event)
        child = event.child()
        if not isinstance(child, ContextualAction):
            return

        match event.type():
            case QEvent.Type.ChildAdded:
                child.changed.connect(self.contextChanged)
                self.contextChanged()

            case QEvent.Type.ChildRemoved:
                child.changed.disconnect(self.contextChanged)
                self.contextChanged()

    def setEnabled(self, enabled: bool) -> None:
        self._enabled = enabled
        self.contextChanged()

    def isEnabled(self) -> bool:
        return self._enabled

    def contextuallyEnabled(self) -> bool:
        return super().isEnabled()

    def addContextual(
        self,
        text: str = "",
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        shortcut: QKeySequence | None = None,
        icon: QIcon | None = None,
        tooltip: str = "",
    ) -> ContextualAction:
        return ContextualAction(text, modifiers, shortcut, tooltip, icon, self)

    def getContextual(self) -> Sequence[ContextualAction]:
        return self.findChildren(ContextualAction, options=Qt.FindChildOption.FindDirectChildrenOnly)

    def queryContextual(self, modifiers: Qt.KeyboardModifier) -> ContextualAction | None:
        for child in reversed(self.getContextual()):
            if child.query(modifiers):
                return child

        return None

    def activeContextual(self) -> ContextualAction | None:
        return self.queryContextual(MAIN_WINDOW.keyboardModifiers)

    @Slot()
    def contextChanged(self) -> None:
        if (current := self.activeContextual()) is None:
            return

        super().setEnabled(self._enabled and current.enabled())
        self.setText(current.text())
        self.setIcon(current.icon())
        self.setShortcut(current.shortcut())

        tooltips = list(chain(
            (current.getDisplayToolTip(),),
            (
                action.getDisplayToolTip()
                for action in self.getContextual()
                if action.visible() and action is not current
            )
        ))

        sep = "\n\n" if any("\n" in tooltip for tooltip in tooltips) else "\n"
        self.setToolTip(sep.join(tooltips))

    def _clicked(self) -> None:
        if (current := self.activeContextual()) is not None:
            current.trigger()

class ColumnLayout(QLayout):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._items: list[QLayoutItem] = []
        self._h_spacing: int = -1
        self._v_spacing: int = -1
        self._cols_hint: int = 0

    def spacing(self) -> int:
        h = self.horizontalSpacing()
        v = self.verticalSpacing()
        return h if h == v and h >= 0 else -1

    def setSpacing(self, spacing: int) -> None:
        self.setHorizontalSpacing(spacing)
        self.setVerticalSpacing(spacing)

    def horizontalSpacing(self) -> int:
        if self._h_spacing >= 0:
            return self._h_spacing

        return self._parentSpacing(QStyle.PixelMetric.PM_LayoutHorizontalSpacing)

    def setHorizontalSpacing(self, spacing: int) -> None:
        self._h_spacing = spacing if spacing >= 0 else -1

    def verticalSpacing(self) -> int:
        if self._v_spacing >= 0:
            return self._v_spacing

        return self._parentSpacing(QStyle.PixelMetric.PM_LayoutVerticalSpacing)

    def setVerticalSpacing(self, spacing: int) -> None:
        self._v_spacing = spacing if spacing >= 0 else -1

    def columnsHint(self) -> int:
        return self._cols_hint

    def setColumnsHint(self, columns: int) -> None:
        self._cols_hint = max(columns, 0)

    def _parentSpacing(self, pm: QStyle.PixelMetric) -> int:
        if (parent := self.parent()) is None:
            return -1

        if isinstance(parent, QWidget):
            return parent.style().pixelMetric(pm, None, parent)
        elif isinstance(parent, QLayout):
            return parent.spacing()
        else:
            raise TypeError(parent.__class__.__name__)

    def count(self) -> int:
        return len(self._items)

    def addItem(self, item: QLayoutItem) -> None:
        self._items.append(item)

    def itemAt(self, index: int) -> QLayoutItem | None:
        try:
            return self._items[index]
        except IndexError:
            return None

    def takeAt(self, index: int) -> QLayoutItem | None:
        try:
            return self._items.pop(index)
        except IndexError:
            return None

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        sizes = self._getItemSizes()
        if not sizes:
            return 0

        return self._doLayout(width, sizes)[0]

    def sizeHint(self) -> QSize:
        sizes = self._getItemSizes()
        if not sizes:
            return QSize(0, 0)

        cols_hint = min(self._cols_hint, len(sizes)) if self._cols_hint else len(sizes)

        return self._doColumnLayout(
            cols_hint, sizes,
            self.contentsMargins(),
            self.horizontalSpacing(),
            self.verticalSpacing()
        )[0]

    def minimumSize(self) -> QSize:
        sizes = self._getItemSizes()
        if not sizes:
            return QSize(0, 0)

        margins = self.contentsMargins()
        hsp = self.horizontalSpacing()
        vsp = self.verticalSpacing()

        h = self._doColumnLayout(
            len(sizes), sizes,
            margins, hsp, vsp
        )[0].height()

        w = self._doColumnLayout(
            1, sizes,
            margins, hsp, vsp
        )[0].width()

        return QSize(w, h)

    def setGeometry(self, rect: QRect) -> None:
        super().setGeometry(rect)

        sizes = self._getItemSizes()
        if not sizes:
            return

        top_left = rect.topLeft()
        for (item, isz), irect in zip(sizes, self._doLayout(rect.width(), sizes)[1]):
            irect.translate(top_left)

            exp = item.expandingDirections()
            isz = QSize(
                irect.width() if Qt.Orientation.Horizontal in exp else isz.width(),
                irect.height() if Qt.Orientation.Vertical in exp else isz.height(),
            )

            irect = QStyle.alignedRect(
                Qt.LayoutDirection.LeftToRight,
                item.alignment(),
                isz, irect
            )

            item.setGeometry(irect)

    def _getItemSizes(self) -> list[tuple[QLayoutItem, QSize]]:
        sizes: list[tuple[QLayoutItem, QSize]] = []
        for item in self._items:
            if not item.isEmpty():
                size = item.sizeHint()
                assert size.isValid()

                sizes.append((item, size))

        return sizes

    def _doLayout(self, width: int, sizes: list[tuple[QLayoutItem, QSize]]) -> tuple[int, list[QRect]]:
        margins = self.contentsMargins()
        hsp = self.horizontalSpacing()
        vsp = self.verticalSpacing()

        max_cols = 0
        x = margins.left() + margins.right() - hsp
        for _, size in sizes:
            x += hsp + size.width()
            if x > width:
                break

            max_cols += 1

        for cols in range(max_cols, 1, -1):
            sz, layout = self._doColumnLayout(cols, sizes, margins, hsp, vsp)
            if sz.width() <= width:
                return sz.height(), layout

        sz, layout = self._doColumnLayout(1, sizes, margins, hsp, vsp)
        return sz.height(), layout

    def _doColumnLayout(
        self,
        n_cols: int,
        sizes: list[tuple[QLayoutItem, QSize]],
        margins: QMargins, hsp: int, vsp: int,
    ) -> tuple[QSize, list[QRect]]:
        n_rows = (len(sizes) + n_cols - 1) // n_cols
        heights = [0 for _ in range(n_rows)]
        widths = [0 for _ in range(n_cols)]

        for idx, (_, sz) in enumerate(sizes):
            row, col = divmod(idx, n_cols)
            heights[row] = max(heights[row], sz.height())
            widths[col] = max(widths[col], sz.width())

        layout: list[QRect] = []

        y = margins.top() - vsp
        for height in heights:
            y += vsp

            x = margins.left() - hsp
            for width in widths:
                x += hsp

                if len(layout) < len(sizes):
                    layout.append(QRect(x, y, width, height))

                x += width

            y += height

            if len(layout) == len(sizes):
                return QSize(x + margins.right(), y + margins.bottom()), layout

        raise AssertionError

class PlainTextEdit(QPlainTextEdit):
    textDropped = Signal()

    def __init__(self) -> None:
        super().__init__()

        self.setAcceptDrops(True)
        self.setTabChangesFocus(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()

            if len(urls) != 1:
                return

            if not (path := urls[0].toLocalFile()):
                return

            if not path.endswith(".txt"):
                return
        elif not event.mimeData().hasText():
            return

        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()

            if len(urls) != 1:
                return

            if not (path := urls[0].toLocalFile()):
                return

            if not path.endswith(".txt"):
                return

            try:
                with open(path, "r", encoding="utf-8") as file:
                    text = file.read()
            except Exception as ex:
                QMessageBox.warning(self, "Failed to Open", f"[{ex.__class__.__name__}] {ex}")
                return
        elif event.mimeData().hasText():
            text = event.mimeData().text()
        else:
            return

        self.setPlainText(text)
        self.textDropped.emit()
        event.acceptProposedAction()

class SingleDropTarget(QWidget):
    imageDropped = Signal(Image, int, str, str)

    def __init__(self) -> None:
        super().__init__()

        self._model: Hydra
        self._seqlen_sb: QSpinBox

    def bind(self, model: Hydra, seqlen_sb: QSpinBox) -> None:
        self._model = model
        self._seqlen_sb = seqlen_sb
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasImage():
            event.acceptProposedAction()
        elif event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) != 1:
                return

            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        seqlen = self._seqlen_sb.value()

        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) != 1:
                return

            path = urls[0].toLocalFile()
            if not path:
                url = urls[0].toString()
                try:
                    resp = requests.get(url, timeout=10)
                    resp.raise_for_status()

                    img = self._model.open_image(resp.content, seqlen)
                except Exception as ex:
                    QMessageBox.warning(self, "Failed to Download", f"[{ex.__class__.__name__}] {ex}")
                    return

                event.acceptProposedAction()
                self.imageDropped.emit(img, seqlen, "", url)
                return

            if os.path.isdir(path):
                paths = list(paths_iter((path,), False))
                if not paths:
                    QMessageBox.warning(self, "Empty Folder", f"No images found in {path}.")
                    return

                path = min(paths)

            try:
                img = self._model.open_image(path, seqlen)
            except Exception as ex:
                QMessageBox.warning(self, "Failed to Open", f"[{ex.__class__.__name__}] {ex}")
                return

            event.acceptProposedAction()
            self.imageDropped.emit(img, seqlen, path, "")
            return

        if event.mimeData().hasImage():
            try:
                qimg: QImage = event.mimeData().imageData()
                qimg.convertToColorSpace(QColorSpace.NamedColorSpace.SRgb, QImage.Format.Format_RGBA8888)
                img = Image.new_from_memory(qimg.constBits(), qimg.width(), qimg.height(), 4, "uchar")
                img = self._model.open_image(img, seqlen)
            except Exception as ex:
                QMessageBox.warning(self, "Failed to Convert", f"[{ex.__class__.__name__}] {ex}")
                return

            event.acceptProposedAction()
            self.imageDropped.emit(img, seqlen, "", "")
            return

class BulkDropTarget(QWidget):
    def __init__(self) -> None:
        super().__init__()

        self._target: QLineEdit | None = None

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if not event.mimeData().hasUrls():
            return

        urls = event.mimeData().urls()
        if len(urls) != 1:
            return

        if not (path := urls[0].toLocalFile()):
            return

        event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        if not event.mimeData().hasUrls():
            return

        urls = event.mimeData().urls()
        if len(urls) != 1:
            return

        if not (path := urls[0].toLocalFile()):
            return

        if not os.path.isdir(path):
            path = os.path.dirname(path)

        if self._target is not None:
            self._target.setText(path)
            event.acceptProposedAction()

    def setDropTarget(self, target: QLineEdit) -> None:
        self._target = target
        self.setAcceptDrops(True)

TagEditItem: TypeAlias = tuple[tuple[int, float, str], QCheckBox, "StatsLabel", QWidget | None]

class TagsEdit(QPlainTextEdit):
    addExclusion = Signal(str)
    visualizeTag = Signal(str)
    showTagInfo = Signal(str)

    def __init__(self, accuracy_bar: QProgressBar) -> None:
        super().__init__()

        self.statsWidget = QWidget()
        self._layout = QGridLayout(self.statsWidget)
        self._layout.setVerticalSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self._layout.setColumnStretch(2, 1)
        self._layout.setHorizontalSpacing(self._layout.horizontalSpacing() * 2)
        self._layout.setVerticalSpacing(1)

        self._rewriter: Rewriter | None = None
        self._items: dict[str, TagEditItem] = {}
        self._labels: dict[str, tuple[Label, float]] = {}
        self._fuzzy: dict[str, str] = {}
        self._suggested: dict[str, list[str]] = {}

        self._accuracy_bar = accuracy_bar

        self._header1 = QLabel("Manual Tags")
        self._header1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._header2 = QLabel("\nIdentified Tags")
        self._header2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.setAcceptDrops(False)
        self.setTabChangesFocus(True)
        self.textChanged.connect(self._textChanged)

    def setItems(
        self,
        outputs: Iterable[tuple[Label, float]],
        labels: dict[str, float],
        rewriter: Rewriter,
    ) -> None:
        self._rewriter = rewriter

        for label_obj, prob in outputs:
            self._labels[rewriter[label_obj.label]] = (
                label_obj, labels.get(label_obj.label, prob)
            )

        for label, prob in labels.items():
            rewritten = rewriter[label]

            ck = self._makeToggle(rewritten)
            self._items[rewritten] = (
                (2, -prob, rewritten),
                ck,
                StatsLabel(
                    self, rewritten, self._labels[rewritten][0],
                    linked=ck, soft_link=False,
                ),
                self._makeBar(prob)
            )

        self._accuracy_bar.setEnabled(True)
        self._accuracy_bar.setTextVisible(True)

    def clearItems(self) -> None:
        self._rewriter = None

        self._items.clear()
        self._labels.clear()
        self._fuzzy.clear()
        self._suggested.clear()

        for idx in range(self._layout.count() - 1, -1, -1):
            item = self._layout.takeAt(idx).widget() # type: ignore[union-attr]
            if (
                item is not self._header1 and
                item is not self._header2
            ):
                item.deleteLater() # type: ignore[union-attr]

        self.blockSignals(True)
        self.setPlainText("")
        self.blockSignals(False)

        self._accuracy_bar.setEnabled(False)
        self._accuracy_bar.setTextVisible(False)
        self._accuracy_bar.setValue(0)

    def cleanupItems(
        self, *,
        clear_invalid: bool = False,
        ensure_implied: bool = False,
        sort: bool = False
    ) -> None:
        if clear_invalid:
            for text in [
                rw
                for (o, _, rw), _, _, _ in self._items.values()
                if o == 0
            ]:
                self._clearItem(text)

        if ensure_implied:
            for text in list(self._items.keys()):
                self._ensureConsequents(text)

        if sort:
            self._items = {
                text: item
                for text, item in sorted(
                    self._items.items(),
                    key=lambda item: item[1][0][1:]
                )
            }

        if clear_invalid or ensure_implied:
            self._updateItems()

        self._updateText()

    def getItems(self) -> Iterable[tuple[str, Label | None, float, bool]]:
        for (_, _, text), cb, _, _ in self._items.values():
            if (label_info := self._labels.get(text)) is not None:
                yield (text, label_info[0], label_info[1], cb.isChecked())
            else:
                yield (text, None, 0.0, cb.isChecked())

    def _shouldRecurse(self) -> bool:
        return Qt.KeyboardModifier.ControlModifier in QGuiApplication.queryKeyboardModifiers()

    def _makeToggle(self, text: str) -> QCheckBox:
        ck = QCheckBox()
        ck.setChecked(True)
        ck.setToolTip("Toggle Tag")

        @connect(ck.toggled)
        def _toggled(checked: bool) -> None:
            recurse = self._shouldRecurse()
            if checked:
                self._ensureItem(text, recurse)
            else:
                self._clearItem(text, recurse)

            if recurse:
                self._updateItems()
            elif checked:
                self._accuracy_bar.setValue(self._accuracy_bar.value() + 1)
            else:
                self._accuracy_bar.setValue(self._accuracy_bar.value() - 1)

            self._updateText()

        return ck

    def _makeRemove(self, text: str, is_valid: bool) -> QCheckBox:
        ck = QCheckBox()
        ck.setChecked(True)
        ck.setToolTip("Remove Tag")

        @connect(ck.toggled)
        def _toggled(checked: bool) -> None:
            assert not checked

            self._clearItem(text, is_valid and self._shouldRecurse())
            self._updateItems()
            self._updateText()

        return ck

    def _getSuggestions(self, text: str) -> list[str]:
        if not self._fuzzy:
            for rewritten, (label_obj, _) in self._labels.items():
                self._fuzzy[self._canonicalize(rewritten)] = rewritten
                self._fuzzy[label_obj.label] = rewritten

        text = self._canonicalize(text)
        if text in self._labels:
            return [(text)]

        if (prior := self._suggested.get(text)) is not None:
            return prior

        options = [
            self._fuzzy[option]
            for option in difflib.get_close_matches(text, self._fuzzy.keys(), n=5)
        ]
        if not options:
            return options

        first = options.pop(0)

        options.sort(key=lambda item: (-self._labels[item][1], item))
        if len(options) > 2:
            options = options[:2]

        options.insert(0, first)

        self._suggested[text] = options
        return options

    def _makeSuggestion(self, orig: str, sugg: str) -> QPushButton:
        label, prob = self._labels[sugg]

        btn = QPushButton(f"{sugg} {int(round(prob * 100))}%")

        @connect(btn.clicked)
        def _clicked() -> None:
            self._clearItem(orig)
            self._ensureItem(sugg, self._shouldRecurse())

            self._updateItems()
            self._updateText()

        return btn

    def _ensureItem(self, text: str, recurse: bool = False) -> None:
        assert self._rewriter is not None

        if (existing := self._items.get(text)) is not None:
            _, ck, lbl, bar = existing

            _set_checked(existing[1], True)

            if isinstance(existing[3], QProgressBar):
                existing[3].setEnabled(True)
        elif (label_info := self._labels.get(text)) is not None:
            label_obj, prob = label_info
            ck = self._makeRemove(text, True)
            self._items[text] = (
                (1, -prob, text),
                ck,
                StatsLabel(self, text, label_obj, linked=ck),
                self._makeBar(prob)
            )
        else:
            ck = self._makeRemove(text, False)
            self._items[text] = (
                (0, 0.0, text),
                ck,
                StatsLabel(self, text, None, linked=ck),
                self._makeSuggestions(text)
            )

        if recurse:
            self._ensureConsequents(text)

    def _ensureConsequents(self, text: str) -> None:
        if (label_info := self._labels.get(text)) is None:
            return

        assert self._rewriter is not None

        for implied in label_info[0].implies:
            self._ensureItem(self._rewriter[implied], True)

    def _clearItem(self, text: str, recurse: bool = False) -> bool:
        retained = False

        if (existing := self._items.get(text)) is not None:
            ((o, _, _), ck, lbl, bar) = existing

            if o == 2:
                retained = True
                _set_checked(ck, False)
                bar.setEnabled(False) # type: ignore[union-attr]
            else:
                self._items.pop(text, None)

                self._layout.removeWidget(ck)
                ck.deleteLater()

                self._layout.removeWidget(lbl)
                lbl.deleteLater()

                if bar is not None:
                    self._layout.removeWidget(bar)
                    bar.deleteLater()

        if recurse:
            self._clearAntecedents(text)

        return retained

    def _clearAntecedents(self, text: str) -> None:
        if (label_info := self._labels.get(text)) is None:
            return

        assert self._rewriter is not None

        for implied in label_info[0].implied_by:
            self._clearItem(self._rewriter[implied], True)

    def _makeSuggestions(self, text: str) -> QWidget | None:
        options = self._getSuggestions(text)
        if not options:
            return None

        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        for option in options:
            layout.addWidget(self._makeSuggestion(text, option))

        layout.addStretch()
        return container

    @staticmethod
    def _makeBar(prob: float) -> QProgressBar:
        bar = QProgressBar()
        bar.sizePolicy().setVerticalPolicy(QSizePolicy.Policy.Minimum)
        bar.setFixedHeight(bar.fontMetrics().height() + 2)

        bar.setMaximum(1000)
        bar.setValue(int(round(prob * 1000)))
        bar.setTextVisible(True)

        return bar

    def _textChanged(self) -> None:
        if self._rewriter is None:
            return

        texts = list(self._rewriter.unjoin(self.toPlainText()))
        retained: list[str] = []

        clear = set(self._items.keys())
        clear.difference_update(texts)

        for text in clear:
            if self._clearItem(text):
                retained.append(text)

        for text in texts:
            self._ensureItem(text)

        # rearrange in text order
        self._items = {
            text: existing
            for text in chain(texts, retained)
            if (existing := self._items.pop(text, None)) is not None
        }
        self._updateItems()

    def _updateText(self) -> None:
        assert self._rewriter is not None

        self.blockSignals(True)
        self.setPlainText(self._rewriter.separator.join(
            rw for ((_, _, rw), ck, _, _) in self._items.values()
            if ck.isChecked()
        ))
        self.blockSignals(False)

    def _updateItems(self) -> None:
        for idx in range(self._layout.count() - 1, -1, -1):
            self._layout.takeAt(idx)

        self._header1.setVisible(False)
        self._header2.setVisible(False)

        n_total = 0
        n_accurate = 0

        shift = 0
        prev_o = -1
        for idx, ((o, _, _), ck, lbl, bar) in enumerate(sorted(
            self._items.values(),
            key=lambda item: item[0]
        )):
            if o in (0, 1) and prev_o == -1:
                self._header1.setVisible(True)
                self._layout.addWidget(self._header1, idx + shift, 0, 1, 2)
                shift += 1
            elif o == 2 and prev_o in (0, 1):
                self._header2.setVisible(True)
                self._layout.addWidget(self._header2, idx + shift, 0, 1, 2)
                shift += 1

            if o == 1:
                n_total += 1
            elif o == 2:
                n_total += 1
                if ck.isChecked():
                    n_accurate += 1

            self._layout.addWidget(ck, idx + shift, 0)
            self._layout.addWidget(lbl, idx + shift, 1)
            if bar is not None:
                self._layout.addWidget(bar, idx + shift, 2)

            prev_o = o

        self._accuracy_bar.setMaximum(n_total)
        self._accuracy_bar.setValue(n_accurate)

    @staticmethod
    def _canonicalize(text: str) -> str:
        return text.lower().replace(" ", "_").replace("\\", "")

class LiveStats(QWidget):
    def __init__(self, editor: TagsEdit) -> None:
        super().__init__()

        self._editor = editor
        self._items: dict[str, tuple[QLabel, "StatsLabel", QProgressBar]] = {}
        self._labels: dict[str, tuple[str, Label]] = {}
        self._count = 0

        self._lock = Lock()
        self._added = 0
        self._counter = Counter[str]()
        self._pending = Counter[str]()

        self._layout = QGridLayout(self)
        self._layout.setVerticalSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        self._layout.setColumnStretch(2, 1)
        self._layout.setHorizontalSpacing(self._layout.horizontalSpacing() * 2)
        self._layout.setVerticalSpacing(1)

    def beginResults(self, rewritten_labels: Iterable[tuple[str, Label]]) -> None:
        self._labels = {
            label[1].label: label
            for label in rewritten_labels
        }

    def endResults(self) -> None:
        self._count = 0
        self._items.clear()
        self._labels.clear()
        self._counter.clear()

    def addResult(self, result: Iterable[str]) -> None:
        with self._lock:
            dirtied = self._added == 0
            self._added += 1
            self._pending.update(result)

        if dirtied:
            QMetaObject.invokeMethod(self, "_update", Qt.ConnectionType.QueuedConnection)

    @Slot()
    def _update(self) -> None:
        new_pending = Counter[str]()
        with self._lock:
            added = self._added
            self._added = 0
            pending = self._pending
            self._pending = new_pending

        del new_pending

        self._count += added
        self._counter.update(pending)

        self.setUpdatesEnabled(False)

        for _, _, bar in self._items.values():
            bar.setMaximum(self._count)

        for label, added in pending.items():
            if (existing := self._items.get(label)) is not None:
                count = self._counter[label]
                existing[2].setValue(count)
                existing[0].setText(str(count))
            else:
                bar = QProgressBar()
                bar.sizePolicy().setVerticalPolicy(QSizePolicy.Policy.Minimum)
                bar.setFixedHeight(self.fontMetrics().height() + 2)

                bar.setMaximum(self._count)
                bar.setValue(added)
                bar.setTextVisible(True)
                bar.setFormat("%v/%m (%p%)")

                self._items[label] = (
                    QLabel(str(added)),
                    StatsLabel(self._editor, *self._labels[label]),
                    bar
                )

        for idx in range(self._layout.count() - 1, -1, -1):
            self._layout.takeAt(idx)

        for idx, (label, (cnt, lbl, bar)) in enumerate(sorted(
            self._items.items(),
            key=lambda item: (-self._counter[item[0]], self._labels[item[0]][0])
        )):
            self._layout.addWidget(cnt, idx, 0)
            self._layout.addWidget(lbl, idx, 1)
            self._layout.addWidget(bar, idx, 2)

        self.setUpdatesEnabled(True)

    def clearResults(self) -> None:
        for idx in range(self._layout.count() - 1, -1, -1):
            self._layout.takeAt(idx).widget().deleteLater() # type: ignore[union-attr]

class StatsLabel(QLabel):
    def __init__(
        self,
        editor: TagsEdit,
        text: str,
        label: Label | None,
        *,
        linked: QCheckBox | None = None,
        soft_link: bool = True,
    ):
        super().__init__(text)
        self.setTextFormat(Qt.TextFormat.PlainText)

        self._text = text
        self._label = label
        self._editor = editor
        self._linked = linked
        self._soft_link = soft_link

        if (color := CATEGORY_COLORS.get(label.category if label is not None else "invalid")) is not None:
            palette = self.palette()
            palette.setColor(QPalette.ColorRole.WindowText, color)
            self.setPalette(palette)

        if label:
            self.setToolTip(f"{label.category}: {label}")
        else:
            self.setToolTip(f"unknown: {text}")

    def mousePressEvent(self, event: QMouseEvent) -> None:
        super().mousePressEvent(event)

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if Qt.KeyboardModifier.ControlModifier in event.modifiers():
            if self._linked is not None:
                if Qt.KeyboardModifier.ShiftModifier in event.modifiers():
                    pass
                else:
                    self._linked.toggle()
        elif Qt.KeyboardModifier.AltModifier in event.modifiers():
            if Qt.KeyboardModifier.ShiftModifier in event.modifiers():
                pass
            elif self._linked is not None and self._label is not None:
                self._editor.visualizeTag.emit(self._label.label)
        elif Qt.KeyboardModifier.ShiftModifier in event.modifiers():
            if self._linked is not None and self._label is not None:
                self._editor._ensureConsequents(self._text)
        elif self._linked is not None and not self._soft_link:
            self._linked.toggle()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        super().mouseDoubleClickEvent(event)

        if event.button() != Qt.MouseButton.LeftButton:
            return

        if self._linked is not None and self._soft_link:
            self._linked.toggle()

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:
        menu = QMenu(self)
        menu.addSection(self._text)

        if self._linked is not None:
            if self._soft_link:
                toggle_action = QAction("Remove &Tag [DoubleClick]")
            else:
                toggle_action = QAction("&Toggle Tag [Click]")

            menu.addAction(toggle_action)

            @connect(toggle_action.triggered)
            def _toggle_action() -> None:
                self._linked.toggle() # type: ignore[union-attr]

            if self._label is not None:
                impls_action = QAction("&Implication-Aware Toggle [Ctrl+Click]")
                menu.addAction(impls_action)

                @connect(impls_action.triggered)
                def _impls_action() -> None:
                    if self._linked.isChecked(): # type: ignore[union-attr]
                        self._editor._clearItem(self._text, True)
                    else:
                        self._editor._ensureItem(self._text, True)

                consequents_action = QAction("&Add Implied Tags [Shift+Click]")
                consequents_action.setEnabled(bool(self._label.implies))
                menu.addAction(consequents_action)

                @connect(consequents_action.triggered)
                def _consequents_action() -> None:
                    self._editor._ensureConsequents(self.text())

                if self._label.implied_by:
                    antecedents_action = QAction("&Remove Implying Tags")
                    menu.addAction(antecedents_action)

                    @connect(antecedents_action.triggered)
                    def _antecedents_action() -> None:
                        self._editor._clearAntecedents(self.text())

        if self._label is not None:
            if self._linked is not None:
                menu.addSeparator()

            excl_action = QAction("&Exclude Tag")
            menu.addAction(excl_action)

            @connect(excl_action.triggered)
            def _excl_action() -> None:
                self._editor.addExclusion.emit(self._label.label) # type: ignore[union-attr]

                if self._linked is not None:
                    self._linked.setChecked(False)

            if self._label.implied_by:
                excl_recursive_action = QAction("E&xclude with Implying Tags")
                menu.addAction(excl_recursive_action)

                @connect(excl_recursive_action.triggered)
                def _excl_recusrive_action() -> None:
                    self._editor.addExclusion.emit(f"~{self._label.label}") # type: ignore[union-attr]

                    if self._linked is not None:
                        self._linked.setChecked(False)
                        self._editor._clearAntecedents(self.text())

                excl_implying_action = QAction("Exclude Implying Tags")
                menu.addAction(excl_implying_action)

                @connect(excl_implying_action.triggered)
                def _excl_implying_action() -> None:
                    self._editor.addExclusion.emit(f"~~{self._label.label}") # type: ignore[union-attr]

                    if self._linked is not None:
                        self._editor._clearAntecedents(self.text())

                menu.addSeparator()

                if self._linked is not None:
                    vis_action = QAction("Attention &Visualization [Alt+Click]")
                    menu.addAction(vis_action)

                    @connect(vis_action.triggered)
                    def _vis_action() -> None:
                        self._editor.visualizeTag.emit(self._label.label) # type: ignore[union-attr]

                info_action = QAction("Show in &Model Info")
                menu.addAction(info_action)

                @connect(info_action.triggered)
                def _info_action() -> None:
                    self._editor.showTagInfo.emit(self._label.label) # type: ignore[union-attr]

        menu.exec(event.globalPos())

class ExistingPathDialog(QFileDialog):
    def __init__(
        self,
        parent: QWidget,
        caption: str = "",
        directory: str = "",
        filter: str = "",
    ) -> None:
        super().__init__(parent, caption, directory, filter)

        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setSupportedSchemes(["file"])

    def accept(self) -> None:
        paths = self.selectedFiles()
        if len(paths) != 1:
            return

        path = paths[0]
        if not os.path.exists(path):
            QMessageBox.warning(
                self,
                self.windowTitle(),
                f"{os.path.basename(path)}\nNot found.\nPlease verify the correct name was given."
            )
            return

        self.filesSelected.emit(paths)
        self.fileSelected.emit(path)
        QDialog.accept(self)

    @staticmethod
    def getExistingPath(
        parent: QWidget,
        caption: str = "",
        directory: str = "",
        filter: str = "",
    ) -> str | None:
        dialog = ExistingPathDialog(parent, caption, directory, filter)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None

        return dialog.selectedFiles()[0]

class Proxy(QObject):
    def __init__(self):
        super().__init__()

        self._queue = deque[tuple[
            Callable[..., None],
            tuple[Any, ...],
            dict[str, Any]
        ]]()

    @Slot()
    def _execute(self) -> None:
        fn, args, kwargs = self._queue.popleft()
        fn(*args, **kwargs)

    def invoke(self, fn: Callable[P, None], *args: P.args, **kwargs: P.kwargs) -> None:
        self._queue.append((fn, args, kwargs))
        QMetaObject.invokeMethod(self, "_execute", Qt.ConnectionType.QueuedConnection)

def _widget(widget: W, parent: QLayout | QMainWindow | QScrollArea, *layout) -> W:
    if isinstance(parent, QMainWindow):
        assert not layout
        parent.setCentralWidget(widget)
    elif isinstance(parent, QScrollArea):
        assert not layout
        parent.setWidget(widget)
    else:
        parent.addWidget(widget, *layout)

    return widget

def _layout(
    ty: Callable[..., L],
    parent: QLayout | QMainWindow | QScrollArea,
    *layout,
    widget: QWidget | None = None,
    margins: bool = False,
) -> L:
    if isinstance(parent, QLayout) and widget is None:
        inst = ty()
        parent.addLayout(inst, *layout) # type: ignore[attr-defined]
    else:
        inst = ty(_widget(
            widget if widget is not None else QWidget(),
            parent, *layout
        ))

    if not margins:
        inst.setContentsMargins(0, 0, 0, 0)

    return inst

def _set_checked(btn: QAbstractButton, checked: bool) -> None:
    btn.blockSignals(True)
    btn.setChecked(checked)
    btn.blockSignals(False)

def _set_tooltip(btn: QAbstractButton, text: str = "") -> None:
    btn.setToolTip(
        f"[{btn.shortcut().toString(QKeySequence.SequenceFormat.NativeText)}]"
        f"{' ' if text else ''}{text}"
    )

@torch.no_grad()
def main() -> None:
    global MAIN_WINDOW

    #if hasattr(torch.backends, "fp32_precision"):
    #    torch.backends.fp32_precision = "tf32"
    #else:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    default_model  = "JTP-Hydra-3.5.safetensors"

    parser = argparse.ArgumentParser(
        description="Hydra Classifier GUI by Project RedRocket",
        epilog=(
            "\n"
            "Visit https://huggingface.co/RedRocket/Hydra for more information."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )

    group = parser.add_argument_group("model")
    group.add_argument(
        "-M", "--model", default=default_model,
        metavar="PATH",
        help=f"Path to model file. (Default: {default_model})",
    )
    group.add_argument(
        "--metadata", default="./data",
        metavar="PATH",
        help="Metadata directory for legacy JTP-3 models. (Default: ./data)",
    )
    group.add_argument(
        "-e", "--extension", action="append", default=[],
        metavar="PATH",
        help=(
            "Path to extension. May be specified multiple times. "
            "If a directory is specified, all extensions in the specified directory are loaded. "
            "(Default: extensions/<model_name>)"
        ),
    )
    group.add_argument(
        "-E", "--no-default-extensions", action="store_true",
        help="Do not load extensions by default.",
    )

    group = parser.add_argument_group("execution")
    group.add_argument(
        "-d", "--device", default=default_device,
        metavar="TORCH_DEVICE",
        help=f"Torch device. (Default: {default_device})",
    )
    group.add_argument(
        "-c", "--compile", action="store_true",
        help="Compile the model for maximum performance.",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    app = QApplication()
    if platform.system() == "Darwin":
        app.setStyle("macintosh")
    else:
        app.setStyle("Fusion")

    model: Hydra | None = None
    worker: Thread | None = None
    loader: Loader | None = None
    loader_cfg: tuple | None = None
    calibration: Calibration | None = None
    calibration_cfg: tuple | None = None
    rewriter: Rewriter | None = None
    rewriter_cfg: tuple | None = None
    stale_calibration = False

    stop_event = Event()
    workqueue = WorkQueue(name="cpu-worker")

    def _tab(
        parent: QTabWidget,
        title: str,
        widget: QWidget | None = None,
        *,
        enabled: bool = True
    ) -> QGridLayout:
        if widget is None:
            widget = QWidget()

        idx = parent.addTab(widget, title)
        parent.setTabEnabled(idx, enabled)
        return QGridLayout(widget)

    root = MainWindow()
    MAIN_WINDOW = root

    root_layout = _layout(QHBoxLayout, root)

    main_tabs = _widget(QTabWidget(), root_layout, 1)
    config_tabs = _widget(QTabWidget(), root_layout, 1)
    config_tabs.setUsesScrollButtons(False)

    def _saveload(
        parent: QGridLayout, row: int,
        txt: QPlainTextEdit,
        header: str, title: str, tooltip: str,
        *,
        toggle: bool = True,
        validate: Callable[[str, bool], str] | None = None,
    ) -> Callable[[], str]:
        last_path = os.path.join(
            "data" if os.path.isdir("data") else "",
            title.lower().replace(" ", "-") + ".txt"
        )

        layout = _layout(QHBoxLayout, parent, row, 0, 1, 2)

        cb = _widget(QCheckBox(), layout) if toggle else None

        lbl = _widget(QLabel(header), layout, 1)
        lbl.setToolTip(tooltip)

        if cb is not None:
            cb.setChecked(True)

            @connect(cb.toggled)
            def _toggled(checked: bool) -> None:
                lbl.setEnabled(checked)

                if checked and validate is not None:
                    if (msg := validate(txt.toPlainText(), False)):
                        QMessageBox.warning(root, title, msg)

        if validate is not None:
            btn_validate = _widget(QPushButton("Check"), layout)
            btn_validate.setFixedWidth(60)
            btn_validate.setToolTip(f"Check that the list of {title.lower()} is valid and report any unrecognized tags.")

            @connect(btn_validate.clicked)
            def _validate() -> None:
                if (msg := validate(txt.toPlainText(), True)):
                    QMessageBox.warning(root, title, msg)
                else:
                    QMessageBox.information(root, title, f"The list of {title.lower()} is valid.")

            if isinstance(txt, PlainTextEdit):
                @connect(txt.textDropped)
                def _dropped() -> None:
                    if (msg := validate(txt.toPlainText(), False)):
                        QMessageBox.warning(root, title, msg)

        btn_save = _widget(QPushButton("Save…"), layout)
        btn_save.setToolTip(f"Save {title.lower()} to file.")
        btn_save.setFixedWidth(60)

        @connect(btn_save.clicked)
        def _save() -> None:
            nonlocal last_path

            if txt.isReadOnly():
                return

            value = txt.toPlainText().strip()
            if not value:
                return

            path, _ = QFileDialog.getSaveFileName(
                root, f"Save {title}", last_path, "Text (*.txt);;Any (*)"
            )
            if not path:
                return

            last_path = path

            try:
                with open(path, "w", encoding="utf-8") as file:
                    file.write(value)
            except Exception as ex:
                QMessageBox.warning(root, "Failed to Save", f"[{ex.__class__.__name__}] {ex}")
                return

        btn_load = _widget(QPushButton("Load…"), layout)
        btn_load.setFixedWidth(60)
        btn_load.setToolTip(f"Load {title.lower()} from file.")

        @connect(btn_load.clicked)
        def _load() -> None:
            nonlocal last_path

            if txt.isReadOnly():
                return

            path, _ = QFileDialog.getOpenFileName(
                root, f"Load {title}", last_path, "Text (*.txt);;Any (*)"
            )
            if not path:
                return

            last_path = path

            try:
                with open(path, "r", encoding="utf-8") as file:
                    value = file.read()
            except Exception as ex:
                QMessageBox.warning(root, "Failed to Open", f"[{ex.__class__.__name__}] {ex}")
                return

            txt.setPlainText(value)

            if validate is not None and (msg := validate(value, False)):
                QMessageBox.warning(root, title, msg)

        if cb is not None:
            return lambda: txt.toPlainText() if cb.isChecked() else ""

        btn_clear = _widget(QPushButton("Clear"), layout)
        btn_clear.setFixedWidth(60)
        btn_clear.setToolTip(f"Clear {title.lower()}.")

        @connect(btn_clear.clicked)
        def _clear() -> None:
            if not txt.isReadOnly():
                txt.setPlainText("")

        return txt.toPlainText

    # Classification Tab
    cls_layout = _tab(config_tabs, "Classification")
    cls_layout.setRowStretch(6, 1)
    cls_layout.setRowStretch(8, 1)
    cls_layout.setColumnStretch(1, 1)

    metric_cb = _widget(QComboBox(), cls_layout, 0, 0)
    metric_cb.setToolTip(
        "Calibration Metric\n"
        "Confidence — Familiar 0.0 (most speculative tags) to 1.0 (most confident tags) slider.\n"
        "  This corresponds to an F Score between 4.0 and 0.25.\n"
        "F Score — Calibrates the model based on precision (confident tags) and recall (speculative tags).\n"
        "  Values below 1.0 favor precision (fewer tags) and values above 1.0 favor recall (more tags).\n"
        "CSI — Uses the Critical Success Index, an alternative metric similar to F Score."
    )
    metric_cb.addItems(("Confidence", "F Score", "CSI"))
    metric_cb.setCurrentText("Confidence")
    metric_cb_current = metric_cb.currentText()

    metric_arg_layout = _layout(QHBoxLayout, cls_layout, 0, 1)

    metric_arg_sb = _widget(QDoubleSpinBox(), metric_arg_layout, 0)
    metric_arg_sb.setRange(0.0, 1.0)
    metric_arg_sb.setSingleStep(0.01)
    metric_arg_sb.setValue(0.5)

    metric_arg_sl = _widget(QSlider(Qt.Orientation.Horizontal), metric_arg_layout, 1)
    metric_arg_sl.setRange(0, 100)
    metric_arg_sl.setValue(50)

    metric_base = 20**(1/250)

    min_prec_lbl = _widget(QLabel("Threshold:"), cls_layout, 1, 0)
    min_prec_lbl.setToolTip(
        "Per-Tag Minimum Precision\n"
        "Increase this to remove less-accurate tags."
    )

    @connect(metric_cb.currentTextChanged)
    def _metric_cb(val: str) -> None:
        nonlocal metric_cb_current

        arg_val = metric_arg_sb.value()

        if val == "Confidence":
            metric_arg_sb.setRange(0.0, 1.0)
            metric_arg_sl.setRange(0, 100)
            metric_arg_sb.setValue(interval_exp(arg_val, 4.0, clamp=True))
        elif metric_cb_current == "Confidence":
            metric_arg_sb.setRange(0.05, 20.0)
            metric_arg_sl.setRange(-250, 250)
            metric_arg_sb.setValue(log_interval(arg_val, 4.0))

        metric_cb_current = val

    @connect(metric_arg_sb.valueChanged)
    def _metric_arg_sb(val: float) -> None:
        if metric_cb.currentText() == "Confidence":
            val_i = int(round(val * 100.0))
        else:
            val_i = int(round(log(val, metric_base)))

        metric_arg_sl.blockSignals(True)
        metric_arg_sl.setValue(val_i)
        metric_arg_sl.blockSignals(False)

    @connect(metric_arg_sl.valueChanged)
    def _metric_arg_sl(val: int) -> None:
        if metric_cb.currentText() == "Confidence":
            val_f = val / 100.0
        else:
            val_f = metric_base**val

        metric_arg_sb.blockSignals(True)
        metric_arg_sb.setValue(val_f)
        metric_arg_sb.blockSignals(False)

    min_prec_layout = _layout(QHBoxLayout, cls_layout, 1, 1)

    min_prec_sb = _widget(QDoubleSpinBox(), min_prec_layout, 0)
    min_prec_sb.setRange(0.0, 0.99)
    min_prec_sb.setSingleStep(0.01)
    min_prec_sb.setValue(0.1)

    min_prec_sl = _widget(QSlider(Qt.Orientation.Horizontal), min_prec_layout, 1)
    min_prec_sl.setRange(0, 99)
    min_prec_sl.setValue(10)

    @connect(min_prec_sb.valueChanged)
    def _min_prec_sb(val: float) -> None:
        min_prec_sl.blockSignals(True)
        min_prec_sl.setValue(int(round(val * 100.0)))
        min_prec_sl.blockSignals(False)

    @connect(min_prec_sl.valueChanged)
    def _min_prec_sl(val: int) -> None:
        min_prec_sb.blockSignals(True)
        min_prec_sb.setValue(val / 100.0)
        min_prec_sb.blockSignals(False)

    implications_lbl = _widget(QLabel("Implications:"), cls_layout, 2, 0)
    implications_lbl.setToolTip(
        "Implications Mode\n"
        "For the most intuitive results, select inherit to be permissive or enforce-inherit to be strict.\n"
        "To still see the raw model outputs, use preserve or enforce instead.\n"
        "For a more cautious assessment, swap inherit for constrain.\n\n"
        "preserve — Tags are preserved if they are implied by another tag. Probabilities may be inconsistent.\n"
        "inherit — Tags are preserved and inherit the highest probability among the tags that imply them.\n"
        "constrain — Tags are preserved and inherit the lowest probability among the tags they imply.\n"
        "enforce — Tags are removed unless all the tags they imply are present. Probabilities may be inconsistent.\n"
        "remove — Exclude all implied tags.\n"
        "constrain-remove — Combination of constrain followed by remove.\n"
        "enforce-inherit — Combination of enforce followed by inherit.\n"
        "enforce-constrain — Combination of enforce followed by constrain.\n"
        "enforce-remove — Combination of enforce followed by remove.\n"
        "off — Raw model output with no implications applied. Probabilities and tags may be inconsistent."
    )

    implications_cb = _widget(QComboBox(), cls_layout, 2, 1)
    implications_cb.addItems(IMPLICATION_MODES)
    implications_cb.setCurrentText("inherit")

    _widget(QLabel("Categories:"), cls_layout, 3, 0).setAlignment(Qt.AlignmentFlag.AlignBaseline)
    excl_cats_layout = _layout(ColumnLayout, cls_layout, 3, 1)
    excl_cats_layout.setColumnsHint(3)
    excl_cats_layout.setVerticalSpacing(0)
    excl_cat_cks: dict[str, QCheckBox] = {}

    def _validate_excl_tags(text: str, strict: bool) -> str:
        try:
            items = parse_expandable_list(text, strict=strict)
        except Exception as ex:
            return f"[{ex.__class__.__name__}] {ex}"

        if model is None:
            return ""

        labels = set(model.label_names())
        unrecognized = {
            label for label, _ in items
            if label not in labels
        }
        if not unrecognized:
            return ""

        return "The following tags are not recognized:\n" + "\n".join(sorted(unrecognized))

    excl_tags_txt = _widget(PlainTextEdit(), cls_layout, 6, 0, 1, 2)
    excl_tags_txt.setWordWrapMode(QTextOption.WrapMode.WordWrap)
    excl_tags_value = _saveload(
        cls_layout, 5, excl_tags_txt,
        "Exclude Tags:", "Excluded Tags",
        "List of tags to exclude in e621 format, separated by whitespace.\n"
        "Tags prefixed with ~ exclude that tag and all tags which imply it.\n"
        "Tags prefixed with ~~ exclude all tags which imply it, but not the tag itself.\n"
        "Comments using # are supported.",
        validate=_validate_excl_tags
    )

    def _validate_excl_groups(text: str, strict: bool) -> str:
        try:
            groups = ExclusiveGroup.parse_file(text)
        except Exception as ex:
            return f"[{ex.__class__.__name__}] {ex}"

        if not strict or model is None:
            return ""

        labels = set(model.label_names())
        unrecognized = {
            label for label in chain.from_iterable(
                chain(group.group, group.requirements, group.exceptions)
                for group in groups
            )
            if label not in labels
        }
        if not unrecognized:
            return ""

        return "The following tags are not recognized:\n" + "\n".join(sorted(unrecognized))

    excl_groups_txt = _widget(PlainTextEdit(), cls_layout, 8, 0, 1, 2)
    excl_groups_txt.setWordWrapMode(QTextOption.WrapMode.NoWrap)
    excl_groups_txt.setReadOnly(True)
    excl_groups_value = _saveload(
        cls_layout, 7, excl_groups_txt,
        "Exclusive Groups:", "Exclusive Groups",
        "List of mutually exclusive tag groups, one per line, with tags in e621 format.\n"
        "When tags in the same group are predicted, only the highest-scoring one is kept.\n"
        "Each group may begin with a precondition defining the tags that must or must not be present for the group to apply.\n"
        "The precondition ends with ':' and not-present requirements are prefixed with '!'.\n"
        "The groups are applied one-after-another in the order listed and are implication-aware.\n"
        "Comments using # are supported.",
        validate=_validate_excl_groups
    )

    fmt_layout = _tab(config_tabs, "Format")
    fmt_layout.setRowStretch(4, 1)

    prefix_layout = _layout(QHBoxLayout, fmt_layout, 0, 0, 1, 2)
    _widget(QLabel("Prefix:"), prefix_layout, 0)
    prefix_txt = _widget(QLineEdit(), prefix_layout, 1)

    opt_layout = _layout(QGridLayout, fmt_layout, 1, 0, 1, 2)
    opt_layout.setColumnStretch(1, 1)

    spaces_ck = _widget(QCheckBox("Spaces"), opt_layout, 0, 0)
    spaces_ck.setChecked(True)
    _widget(QLabel("example_tag → example tag"), opt_layout, 0, 1)

    prefix_by_ck = _widget(QCheckBox('"by" Prefix'), opt_layout, 1, 0)
    prefix_by_ck.setEnabled(False)
    _widget(QLabel(r"artist → by_artist"), opt_layout, 1, 1)

    escape_ck = _widget(QCheckBox("Prompt Syntax"), opt_layout, 2, 0)
    _widget(QLabel(r"(suffix) → \(suffix\)"), opt_layout, 2, 1)

    shuffle_ck = _widget(QCheckBox("Shuffle Tags"), fmt_layout, 2, 0, 1, 2)

    def _validate_aliases(text: str, strict: bool) -> str:
        if model is None:
            return ""

        try:
            _rewriter((
                text,
                prefix_by_ck.isChecked(),
                spaces_ck.isChecked(),
                escape_ck.isChecked(),
            ))
        except Exception as ex:
            return f"[{ex.__class__.__name__}] {ex}"

        if not strict:
            return ""

        labels = set(model.label_names())
        unrecognized = {
            label for label, _ in parse_aliases(text)
            if label not in labels
        }
        if not unrecognized:
            return ""

        return f"The following tags are not recognized:\n" + "\n".join(sorted(unrecognized))

    aliases_txt = _widget(PlainTextEdit(), fmt_layout, 4, 0, 1, 2)
    aliases_txt.setWordWrapMode(QTextOption.WrapMode.NoWrap)
    aliases_value = _saveload(
        fmt_layout, 3, aliases_txt,
        "Tag Aliases:", "Aliases",
        "List of tag aliases, one per line.\n"
        "The e621-format model tag comes first, followed by whitespace, and then the e621-format output tag.\n"
        "The output tag must not be a tag already output by the model, unless that tag is also aliased.\n"
        "Comments using # are supported.",
        validate=_validate_aliases,
    )

    # Settings Tab
    config_layout = _tab(config_tabs, "Settings")
    config_layout.setColumnStretch(2, 1)
    config_layout.setRowStretch(5, 1)

    _widget(QLabel("Batch Size:"), config_layout, 0, 0)
    batch_sb = _widget(QSpinBox(), config_layout, 0, 1)
    batch_sb.setRange(1, 128)
    batch_sb.setValue(4)

    _widget(QLabel("Loader Workers:"), config_layout, 1, 0)
    workers_sb = _widget(QSpinBox(), config_layout, 1, 1)
    workers_sb.setRange(-2, 32)
    workers_sb.setValue(min(int(ceil((os.cpu_count() or 1) / 3)), 16))

    _widget(QLabel("NaFlex Seqlen:"), config_layout, 2, 0)
    seqlen_sb = _widget(QSpinBox(), config_layout, 2, 1)
    seqlen_sb.setRange(64, 2048)
    seqlen_sb.setValue(1024)

    varlen_ck = _widget(QCheckBox("Varlen Attention"), config_layout, 3, 0, 1, 2)
    varlen_ck.setEnabled(HAS_VARLEN_ATTN)

    shm_ck = _widget(QCheckBox("Shared Memory"), config_layout, 4, 0, 1, 2)
    shm_ck.setChecked(True)

    help_layout = _layout(QHBoxLayout, config_layout, 6, 0, 1, 2)
    about_qt_btn = _widget(QPushButton("About Qt"), help_layout, 0)
    _widget(QLabel("Apache 2.0 Licensed"), help_layout, 0)
    help_layout.addStretch()

    # Info Tab
    info_layout = _tab(config_tabs, "Model Info", enabled=False)
    info_layout.setRowStretch(1, 1)

    info_header = _layout(QHBoxLayout, info_layout, 0, 0)

    info_model_lbl = _widget(QLabel(), info_header)
    info_model_lbl.setTextFormat(Qt.TextFormat.PlainText)

    info_header.addSpacing(5)
    info_header.addStretch()
    info_header.addSpacing(5)

    info_status_lbl = _widget(QLabel(), info_header)
    info_status_lbl.setTextFormat(Qt.TextFormat.PlainText)

    info_tags_tr = _widget(QTreeWidget(), info_layout, 1, 0)
    info_tags_tr.setHeaderHidden(True)
    info_tags_tr.setColumnCount(2)
    info_tags_tr.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    info_tags_tr.header().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)

    tag_info_items: dict[str, QTreeWidgetItem] = {}

    # Stats Tab
    stats_layout = _tab(config_tabs, "Stats", enabled=False)
    stats_layout.setColumnStretch(0, 0)
    stats_layout.setRowStretch(0, 0)

    stats_bulk_scroll = _widget(QScrollArea(), stats_layout, 0, 0)
    stats_bulk_scroll.setWidgetResizable(True)
    stats_bulk_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    stats_bulk_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    stats_single_scroll = _widget(QScrollArea(), stats_layout, 0, 0)
    stats_single_scroll.setWidgetResizable(True)
    stats_single_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
    stats_single_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    stats_single_scroll.setVisible(False)

    stats_single_ctrls_widget = QWidget()
    stats_single_ctrls = _layout(
        QHBoxLayout, stats_layout, 1, 0,
        widget=stats_single_ctrls_widget
    )

    ensure_implied_btn = _widget(QPushButton("&Add Implied"), stats_single_ctrls, 0)
    _set_tooltip(ensure_implied_btn, "Add missing implied tags.")

    csv_btn = _widget(ContextualButton(), stats_single_ctrls, 0)
    csv_copy_action = csv_btn.addContextual("Copy &CSV", shortcut=QKeySequence("Alt+C"))
    csv_save_action = csv_btn.addContextual(
        "Save &CSV",
        modifiers=Qt.KeyboardModifier.ShiftModifier,
        shortcut=QKeySequence("Alt+Shift+C")
    )

    accuracy_bar = _widget(QProgressBar(), stats_single_ctrls, 1)
    accuracy_bar.setFormat("%v/%m")
    accuracy_bar.setEnabled(False)
    accuracy_bar.setTextVisible(False)

    @connect(main_tabs.currentChanged)
    def _main_tab(index: int) -> None:
        stats_bulk_scroll.setVisible(index == 0)
        stats_single_scroll.setVisible(index == 1)
        stats_single_ctrls_widget.setVisible(index == 1)

    # Bulk Tab
    bulk_target = BulkDropTarget()
    bulk_layout = _tab(main_tabs, "Bulk Autotagging", bulk_target)
    bulk_layout.setColumnStretch(0, 1)
    bulk_layout.setRowStretch(3, 1)

    def _browse_dir() -> None:
        path = QFileDialog.getExistingDirectory(
            root, "Select Folder", path_txt.text(),
            QFileDialog.Option.ShowDirsOnly
        )
        if path:
            path_txt.setText(path)

    folder_layout = _layout(QHBoxLayout, bulk_layout, 0, 0)

    _widget(QLabel("Folder:"), folder_layout, 0)

    path_txt = _widget(QLineEdit(), folder_layout, 1)
    bulk_target.setDropTarget(path_txt)

    path_completer = QCompleter()
    path_model = QFileSystemModel()
    path_model.setRootPath("")
    path_model.setFilter(QDir.Filter.Dirs | QDir.Filter.NoDotAndDotDot)
    path_completer.setModel(path_model)
    path_completer.setCompletionMode(QCompleter.CompletionMode.UnfilteredPopupCompletion)
    path_txt.setCompleter(path_completer)

    browse_btn = _widget(QPushButton("…"), folder_layout, 0)
    browse_btn.setFixedWidth(30)
    browse_btn.clicked.connect(_browse_dir)

    recursive_ck = _widget(QCheckBox("Subfolders"), folder_layout, 0)

    bulk_ctrls = _layout(QHBoxLayout, bulk_layout, 1, 0)

    bulk_btn = _widget(QPushButton("▶ Classify"), bulk_ctrls, 0)
    bulk_btn.setFixedWidth(120)
    bulk_btn.setEnabled(False)

    bulk_bar = _widget(QProgressBar(), bulk_ctrls, 1)
    bulk_bar.setValue(0)
    bulk_bar.setTextVisible(False)
    bulk_bar.setFormat("%v/%m")

    _widget(QLabel("Log"), bulk_layout, 2, 0)

    log_txt = _widget(QPlainTextEdit(), bulk_layout, 3, 0)
    log_txt.setFont(QFont("Consolas", 10))
    log_txt.setWordWrapMode(QTextOption.WrapMode.WordWrap)
    log_txt.setReadOnly(True)

    # Single Tab
    single_target = SingleDropTarget()
    single_layout = _tab(main_tabs, "Single Image", single_target, enabled=False)
    single_layout.setColumnStretch(0, 1)
    single_layout.setRowStretch(3, 1)

    single_image = _widget(QLabel("Click Open or drag and drop an image, folder, or link."), single_layout, 0, 0)
    single_image.setAlignment(Qt.AlignmentFlag.AlignCenter)

    open_lbl = _widget(QLabel(""), single_layout, 1, 0)
    open_lbl.setTextFormat(Qt.TextFormat.PlainText)
    open_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)

    single_ctrls = _layout(QHBoxLayout, single_layout, 2, 0)
    open_btn = _widget(QPushButton("Open…"), single_ctrls, 0)
    prev_btn = _widget(QPushButton("←"), single_ctrls, 0)
    next_btn = _widget(QPushButton("→"), single_ctrls, 0)
    rerun_btn = _widget(ContextualButton(), single_ctrls, 0)
    analysis_btn = _widget(ContextualButton(), single_ctrls, 0)
    single_ctrls.addSpacing(5)
    single_ctrls.addStretch()
    single_ctrls.addSpacing(5)
    clean_btn = _widget(ContextualButton(), single_ctrls, 0)
    copy_btn = _widget(QPushButton("Copy"), single_ctrls, 0)
    save_btn = _widget(ContextualButton(), single_ctrls, 0)

    rerun_action = rerun_btn.addContextual(
        "Rerun",
        shortcut=QKeySequence("F2"),
        tooltip="[%s] Run classifier."
    )
    preview_action = rerun_btn.addContextual(
        "Preview",
        modifiers=Qt.KeyboardModifier.ShiftModifier,
        shortcut=QKeySequence("Shift+F2"),
        tooltip="[%s] Run classifier preview."
    )

    analysis_btn.setCheckable(True)
    cam_action = analysis_btn.addContextual(
        "Visualize",
        shortcut=QKeySequence("F3"),
        tooltip="[%s] Attention Visualization",
    )
    pca_action = analysis_btn.addContextual(
        "PCA",
        modifiers=Qt.KeyboardModifier.ShiftModifier,
        shortcut=QKeySequence("Shift+F3"),
        tooltip="[%s] Principal Component Analysis",
    )

    clean_action = clean_btn.addContextual(
        "C&lean",
        shortcut=QKeySequence("Alt+L"),
        tooltip="[%s] Reformat tag list."
    )
    sort_action = clean_btn.addContextual(
        "Sort",
        modifiers=Qt.KeyboardModifier.ShiftModifier,
        shortcut=QKeySequence("Alt+Shift+L"),
        tooltip="[%s] Sort and reformat tag list."
    )

    save_action = save_btn.addContextual("Save", shortcut=QKeySequence("Ctrl+S"))
    saveas_action = save_btn.addContextual(
        "Save As…",
        shortcut=QKeySequence("Ctrl+Shift+S"),
        tooltip="[%s] Save tags to file."
    )
    save_action.setVisible(False)

    open_btn.setShortcut(QKeySequence.StandardKey.Open)
    prev_btn.setShortcut(QKeySequence.StandardKey.Back)
    next_btn.setShortcut(QKeySequence.StandardKey.Forward)
    copy_btn.setShortcut(QKeySequence.StandardKey.Copy)

    _set_tooltip(open_btn, "Open image or folder.")
    _set_tooltip(prev_btn, "Previous image in folder.")
    _set_tooltip(next_btn, "Next image in folder.")
    _set_tooltip(copy_btn, "Copy tags to clipboard.")

    prev_btn.setFixedWidth(30)
    next_btn.setFixedWidth(30)
    open_btn.setFixedWidth(75)
    rerun_btn.setFixedWidth(75)
    analysis_btn.setFixedWidth(75)

    clean_btn.setFixedWidth(60)
    copy_btn.setFixedWidth(60)
    save_btn.setFixedWidth(75)

    prev_btn.setVisible(False)
    next_btn.setVisible(False)
    rerun_btn.setEnabled(False)
    analysis_btn.setEnabled(False)

    single_txt = _widget(TagsEdit(accuracy_bar), single_layout, 3, 0)
    stats_single_scroll.setWidget(single_txt.statsWidget)

    bulk_stats = _widget(LiveStats(single_txt), stats_bulk_scroll)

    proxy = Proxy()

    open_default = ""
    open_path = ""
    save_path = ""

    single_seqlen = 0
    single_input: Tensor | None = None
    single_features: NaFlexFeatures | NaFlexFeaturesVarlen | None = None
    single_output: Tensor | None = None
    single_preview = False
    single_preserved = ""

    def _enableEditActions(enabled: bool) -> None:
        clean_btn.setEnabled(enabled)
        ensure_implied_btn.setEnabled(enabled)

    _enableEditActions(False)

    def _log(msg: str) -> None:
        log_txt.moveCursor(QTextCursor.MoveOperation.End)
        log_txt.insertPlainText(msg)
        log_txt.moveCursor(QTextCursor.MoveOperation.End)

    @torch.no_grad()
    def _startup() -> None:
        nonlocal model

        try:
            proxy.invoke(_log, f"Loading {repr(args.model)} ...")
            model = load_model(args.model, legacy_metadata_dir=args.metadata)
            proxy.invoke(_log, f" {len(model.labels)} tags.\n")

            if (not args.extension and not args.no_default_extensions):
                default_extensions = "extensions/" + os.path.splitext(os.path.basename(args.model))[0]
                if os.path.isdir(default_extensions):
                    args.extension.append(default_extensions)

            if args.extension:
                proxy.invoke(_log, "Loading extensions ...\n")
                for ext in model.load_extensions(Extension.discover(args.extension)):
                    proxy.invoke(_log,
                        f"  {repr(os.path.basename(ext.path))}: "
                        f"{repr(ext.label.label)} ({ext.label.category})\n"
                    )

            if device.type != "cpu":
                proxy.invoke(_log, f"Transferring to device {device} ...")
                model = model.to(device=device)
                proxy.invoke(_log, " done.\n")

            graph = model.ensure_label_graph()
            assert graph is not None

            if args.compile:
                model.compile(mode="max-autotune-no-cudagraphs")
        except Exception as ex:
            print_exc()
            proxy.invoke(_log, f"[{ex.__class__.__name__}] {ex}\n")
            model = None
        else:
            proxy.invoke(_startup_done, graph)

    def _startup_done(graph: dict[str, Label]) -> None:
        assert model is not None

        _log("Ready.\n\n")

        bulk_btn.setEnabled(True)
        main_tabs.setTabEnabled(1, True)

        categories_set = model.get_categories()

        cat_tree_items: dict[str, QTreeWidgetItem] = {}
        cat_counter = Counter[str]()

        for i, category in enumerate(sorted(categories_set)):
            excl_cat_ck = _widget(QCheckBox(category or "unknown"), excl_cats_layout)
            excl_cat_ck.setChecked(category != "rating")

            excl_cats_layout.setAlignment(excl_cat_ck, Qt.AlignmentFlag.AlignBaseline)
            excl_cat_cks[category] = excl_cat_ck

            if category not in ("color", "rating"):
                cat_item = QTreeWidgetItem(info_tags_tr, (category or "unknown",))
                cat_item.setData(0, Qt.ItemDataRole.UserRole, excl_cat_ck)
                cat_tree_items[category] = cat_item

        if "artist" in categories_set:
            prefix_by_ck.setEnabled(True)
            prefix_by_ck.setChecked(True)

        if "color" in categories_set:
            cat_item = QTreeWidgetItem(cat_tree_items["general"], ("<color>",))
            cat_item.setData(0, Qt.ItemDataRole.UserRole, excl_cat_cks["color"])
            cat_tree_items["color"] = cat_item

        if "rating" in categories_set:
            cat_item = QTreeWidgetItem(cat_tree_items["meta"], ("<rating>",))
            cat_item.setData(0, Qt.ItemDataRole.UserRole, excl_cat_cks["rating"])
            cat_tree_items["rating"] = cat_item

        if "species" in categories_set:
            cat_counter["species"] -= 1
            graph["flying_purple_people_eater"] = Label(
                "flying_purple_people_eater", "species", (
                    "1_eye", "1_horn", "flying",
                    "pigeon_toed", "dwarfism",
                    "imminent_vore",
                )
            )

        def _add_implies(parent: QTreeWidgetItem, label: Label) -> set[str]:
            closure = set(label.implies)

            for impl in sorted(label.implies):
                impl_item = QTreeWidgetItem(parent, (impl,))
                closure |= _add_implies(impl_item, graph[impl])

            if closure:
                parent.setText(1, f"({len(closure)})")

            return closure

        def _add_implying(parent: QTreeWidgetItem, label: Label) -> set[str]:
            closure = set(label.implied_by)

            for impl in sorted(label.implied_by):
                impl_item = QTreeWidgetItem(parent, (impl,))
                implying = _add_implying(impl_item, graph[impl])
                if implying:
                    impl_item.setData(0, Qt.ItemDataRole.UserRole, True)

                closure |= implying

            if closure:
                parent.setText(1, f"({len(closure)})")

            return closure

        for _, label in sorted(graph.items()):
            cat_counter[label.category] += 1

            if (subcategory := label.subcategory) is not None:
                cat_counter[subcategory] += 1
                lbl_parent = cat_tree_items[subcategory]
            else:
                lbl_parent = cat_tree_items[label.category]

            lbl_item = QTreeWidgetItem(lbl_parent, (label.label,))

            implies_item = QTreeWidgetItem(("<implies>",))
            if _add_implies(implies_item, label):
                implies_item.setData(0, Qt.ItemDataRole.UserRole, False)
                lbl_item.addChild(implies_item)

            implies_item = QTreeWidgetItem(("<implied by>",))
            if _add_implying(implies_item, label):
                implies_item.setData(0, Qt.ItemDataRole.UserRole, False)
                lbl_item.addChild(implies_item)
                lbl_item.setData(0, Qt.ItemDataRole.UserRole, True)

            if label.label == "flying_purple_people_eater":
                lbl_item.setData(0, Qt.ItemDataRole.UserRole, False)
            else:
                tag_info_items[label.label] = lbl_item

        for category, item in cat_tree_items.items():
            item.setText(1, f"({cat_counter[category]})")

        info_tags_tr.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        @connect(info_tags_tr.customContextMenuRequested)
        def _info_tags_menu(position: QPoint) -> None:
            if (item := info_tags_tr.itemAt(position)) is None:
                return

            data = item.data(0, Qt.ItemDataRole.UserRole)
            if data is False:
                return

            menu = QMenu(info_tags_tr)
            menu.addSection(item.text(0))

            if isinstance(data, QCheckBox):
                act_enable = QAction("&Include Category", menu)
                act_enable.triggered.connect(lambda: data.setChecked(True))
                act_disable = QAction("&Exclude Category", menu)
                act_disable.triggered.connect(lambda: data.setChecked(False))
                menu.addActions((act_enable, act_disable))
            else:
                act_exclude = QAction("Add to &Excluded Tags", menu)
                act_exclude.triggered.connect(lambda: _add_exclusion(item.text(0)))
                menu.addAction(act_exclude)

                if data is True:
                    act_exclude_recursive = QAction("Exclude with Implying Tags", menu)
                    act_exclude_recursive.triggered.connect(lambda: _add_exclusion(f"~{item.text(0)}"))
                    act_exclude_implying = QAction("Exclude Implying Tags", menu)
                    act_exclude_implying.triggered.connect(lambda: _add_exclusion(f"~~{item.text(0)}"))
                    menu.addActions((act_exclude_recursive, act_exclude_implying))
                    menu.addSeparator()

                act_visualize = QAction("Attention &Visualization", menu)
                act_visualize.setEnabled(single_input is not None and single_features is not None)
                act_visualize.triggered.connect(lambda: _visualize_tag(item.text(0)))
                menu.addAction(act_visualize)

            menu.exec(info_tags_tr.viewport().mapToGlobal(position))

        gp = "(?:(?:fe)?male|intersex|ambiguous)"
        fp = "(?:anthro|feral|human(?:oid)?|taur|draconcopode)"
        gender_expr = re.compile(fr"^{gp}/{gp}$")
        form_expr = re.compile(fr"^{fp}_on_{fp}$")

        gender_combos: list[str] = []
        form_combos: list[str] = []
        for label in model.labels:
            if label.category == "general":
                if gender_expr.fullmatch(label.label):
                    gender_combos.append(label.label)
                elif form_expr.fullmatch(label.label):
                    form_combos.append(label.label)

        excl_groups_txt.setReadOnly(False)
        excl_groups_txt.setPlainText(
            "explicit questionable safe\n\n"
            "# caution: will not handle all multi-scene images correctly\n"
            "!comic !multiple_scenes: zero_pictured solo duo trio group\n"
            "solo !duo !trio !group: anthro feral human humanoid taur draconcopode split_form ambiguous_form\n"
            "solo !duo !trio !group: male female intersex ambiguous_gender\n"
            "solo !duo !trio !group: andromorph gynomorph herm maleherm\n"
            "solo !duo !trio !group: horizontal_diphallism vertical_diphallism hemipenes\n"
            "solo !duo !trio !group !multiple_poses !model_sheet: retracted_foreskin partially_retracted_foreskin unretracted_foreskin\n"
            f"duo !trio !group: {' '.join(gender_combos)}\n"
            f"duo !trio !group: {' '.join(form_combos)}\n"
            f"duo !trio !group: interspecies intraspecies\n"
        )

        config_tabs.setTabEnabled(3, True)
        info_model_lbl.setText(model.name)
        single_target.bind(model, seqlen_sb)

    def _calibrate(new_calibration_cfg: tuple[str, float, float], *, log: bool = False) -> None:
        nonlocal calibration, calibration_cfg
        assert model is not None

        if new_calibration_cfg == calibration_cfg:
            return

        match new_calibration_cfg[0]:
            case "Confidence":
                metric_fn = f_score(log_interval(new_calibration_cfg[1], 4.0))
            case "F Score":
                metric_fn = f_score(new_calibration_cfg[1])
            case "CSI":
                metric_fn = csi(new_calibration_cfg[1])
            case _:
                raise AssertionError

        metric_fn = min_precision(metric_fn, new_calibration_cfg[2])

        if log:
            proxy.invoke(_log, "Calibrating ...")

        try:
            calibration = model.calibrate(metric_fn)
            calibration_cfg = new_calibration_cfg
        except:
            if log:
                proxy.invoke(_log, "\n")

            raise

        if log:
            proxy.invoke(_log, " done.\n")

        proxy.invoke(_display_calibration, calibration)

    def _display_calibration(calibration: Calibration) -> None:
        for label, threshold in calibration.values():
            display_threshold = f"{threshold*100:.1f}%" if threshold <= 1.0 else "—"
            tag_info_items[label.label].setText(1, display_threshold)

    def _rewriter(new_rewriter_cfg: tuple[str, bool, bool, bool]) -> None:
        nonlocal rewriter, rewriter_cfg
        assert model is not None

        if new_rewriter_cfg == rewriter_cfg:
            return

        prefixes: dict[str, str] = {}
        if new_rewriter_cfg[1]:
            prefixes["artist"] = "by_"

        try:
            rewriter = Rewriter.create(
                model.labels,
                aliases=new_rewriter_cfg[0],
                prefixes=prefixes,
                spaces=new_rewriter_cfg[2],
                escape=new_rewriter_cfg[3],
            )
            rewriter_cfg = new_rewriter_cfg
        except:
            rewriter = None
            rewrite_cfg = None
            raise

    @torch.no_grad()
    def _bulk_worker(
        *,
        paths: list[str],
        new_rewriter_cfg: tuple[str, bool, bool, bool],
        new_calibration_cfg: tuple[str, float, float],
        new_loader_cfg: tuple[int, int, bool, int],
        varlen_attn: bool,
        implications: str,
        exclude_categories: set[str],
        exclude_tags: str,
        exclusive_groups: str,
        prefix: str,
        shuffle: bool,
    ) -> None:
        assert model is not None
        nonlocal loader, loader_cfg

        n_done = 0
        cleanup_loader= False

        try:
            _rewriter(new_rewriter_cfg)
            _calibrate(new_calibration_cfg, log=True)

            if new_loader_cfg != loader_cfg:
                if loader is not None:
                    proxy.invoke(_log, "Restarting dataloader ...")
                    try:
                        loader.shutdown(wait=True)
                    except:
                        print_exc()
                    finally:
                        loader = None
                        loader_cfg = None

                    proxy.invoke(_log, " done.\n")
                else:
                    proxy.invoke(_log, "Starting dataloader ...")

                try:
                    model.image_config.max_seqlen = new_loader_cfg[3]
                    loader = Loader(
                        queue_depth=new_loader_cfg[0],
                        config=model.image_config,
                        n_workers=new_loader_cfg[1],
                        share_memory=new_loader_cfg[2]
                    )
                    loader_cfg = new_loader_cfg
                except:
                    proxy.invoke(_log, "\n")
                    raise
                else:
                    proxy.invoke(_log, f" {loader.n_workers} worker{'' if loader.n_workers == 1 else 's'}.\n")

            def _on_done(path: str, labels: dict[str, float]) -> None:
                nonlocal n_done
                assert calibration is not None
                assert rewriter is not None

                n_done += 1
                proxy.invoke(_bulk_item_done, "")
                bulk_stats.addResult(labels.keys())

            n_failed = 0
            def _on_error(path: str, ex: Exception) -> None:
                nonlocal n_failed
                n_failed += 1

                msg = f"{repr(path)}: [{ex.__class__.__name__}] {ex}\n"
                print(msg, end="", file=sys.stderr)
                proxy.invoke(_bulk_item_done, msg)

            assert loader is not None
            assert rewriter is not None
            assert calibration is not None

            bulk_stats.beginResults(
                (rewriter[label.label], label)
                for label in model.labels
            )

            cleanup_loader = True

            start = time.perf_counter()
            process_batched(
                model=model,
                loader=loader,
                paths=paths,
                batch_size=loader_cfg[0],
                calibration=calibration,
                implications=implications,
                exclude_categories=exclude_categories,
                exclude_tags=exclude_tags,
                exclusive_groups=exclusive_groups,
                rewrite={
                    "rewriter": rewriter,
                    "prefix": prefix,
                    "shuffle": shuffle,
                },
                varlen_attn=varlen_attn,
                workqueue=workqueue,
                on_done=_on_done,
                on_error=_on_error,
                stop_event=stop_event
            )

            elapsed = time.perf_counter() - start
            proxy.invoke(_log,
                f"Processed: {n_done}/{len(paths)} (errors: {n_failed})\n"
                f"Elapsed:   {elapsed:.1f}s ({n_done / elapsed:.1f}/s)\n\n"
            )
        except Exception as ex:
            print_exc()
            proxy.invoke(_log, f"[{ex.__class__.__name__}] {ex}\n\n")

            if cleanup_loader:
                assert loader is not None
                proxy.invoke(_log, "Stopping dataloader ...")

                try:
                    loader.shutdown(wait=True)
                except:
                    print_exc()
                finally:
                    loader = None
                    loader_cfg = None

                proxy.invoke(_log, " done.\n")

        proxy.invoke(_bulk_worker_done, n_done)

    def _bulk_item_done(msg: str) -> None:
        bulk_bar.setValue(bulk_bar.value() + 1)
        if msg:
            _log(msg)

    def _bulk_worker_done(n_done: int) -> None:
        nonlocal worker
        assert worker is not None

        worker.join()
        worker = None

        bulk_stats.endResults()

        main_tabs.setTabEnabled(1, True)
        bulk_btn.setText("▶ Classify")
        bulk_btn.setEnabled(True)

        if stale_calibration:
            _sync_calibration()

    @connect(bulk_btn.clicked)
    def _bulk_btn() -> None:
        nonlocal worker, open_default

        if worker is not None:
            bulk_btn.setEnabled(False)
            _log("Stopping ...\n")

            stop_event.set()
            return

        if not path_txt.text():
            _browse_dir()

            if not path_txt.text():
                return

        dir_path = path_txt.text()
        paths = list(paths_iter((dir_path,), recursive=recursive_ck.isChecked()))
        if not paths:
            QMessageBox.warning(root, "Cannot Start", "No images found.")
            return

        open_default = dir_path

        main_tabs.setTabEnabled(1, False)
        config_tabs.setTabEnabled(4, True)
        config_tabs.setCurrentIndex(4)

        bulk_bar.setValue(0)
        bulk_bar.setMaximum(len(paths))
        bulk_bar.setTextVisible(True)
        bulk_stats.clearResults()

        bulk_btn.setText("■ Stop")

        _log(f"Classifying {len(paths)} file{'' if len(paths) == 1 else 's'} in {repr(dir_path)}...\n")

        stop_event.clear()
        worker = Thread(target=_bulk_worker, kwargs={
            "paths": paths,
            "new_rewriter_cfg": (
                aliases_value(),
                prefix_by_ck.isChecked(),
                spaces_ck.isChecked(),
                escape_ck.isChecked(),
            ),
            "new_calibration_cfg": (
                metric_cb.currentText(),
                metric_arg_sb.value(),
                min_prec_sb.value()
            ),
            "new_loader_cfg": (
                batch_sb.value(),
                Loader.heuristic_workers(workers_sb.value(), len(paths), batch_sb.value()),
                shm_ck.isChecked(),
                seqlen_sb.value()
            ),
            "varlen_attn": varlen_ck.isChecked(),
            "implications": implications_cb.currentText(),
            "exclude_categories": {
                cat
                for cat, ck in excl_cat_cks.items()
                if not ck.isChecked()
            },
            "exclude_tags": excl_tags_value(),
            "exclusive_groups": excl_groups_value(),
            "prefix": prefix_txt.text(),
            "shuffle": shuffle_ck.isChecked(),
        }, name="bulk-worker", daemon=True)
        worker.start()

    @connect(next_btn.clicked)
    def _next_btn() -> None:
        _next_file(False)

    @connect(prev_btn.clicked)
    def _prev_btn() -> None:
        _next_file(True)

    def _next_file(reverse: bool) -> None:
        assert open_path
        assert model is not None

        op = str.__lt__ if reverse else str.__gt__
        scan_dir, scan_name = os.path.split(open_path)
        next_name = ""
        next_path = ""

        for path in paths_iter((scan_dir,), False):
            name = os.path.basename(path)
            if op(name, scan_name) and (not next_name or op(next_name, name)):
                next_name = name
                next_path = path

        if next_path:
            _open_path(next_path)
        elif reverse:
            QMessageBox.information(root, "At Beginning", "The current file is the first file.")
        else:
            QMessageBox.information(root, "At End", "The current file is the last file.")

    @connect(open_btn.clicked)
    def _open_btn() -> None:
        assert model is not None

        path = ExistingPathDialog.getExistingPath(
            root, f"Open Image", open_default,
            "Images (*.avif *.bmp *.gif *.jpeg *.jpg *.jxl *.png *.tiff *.webp);;Any (*)"
        )
        if path:
            _open_path(path)

    def _open_path(path: str) -> None:
        nonlocal single_seqlen
        assert model is not None

        seqlen = seqlen_sb.value()

        if os.path.isdir(path):
            paths = list(paths_iter((path,), False))
            if not paths:
                QMessageBox.warning(root, "Empty Folder", f"No images found in {path}.")
                return

            path = min(paths)

        try:
            img = model.open_image(path, seqlen)
        except Exception as ex:
            _select_failed_path(path)
            QMessageBox.warning(root, "Failed to Open", f"[{ex.__class__.__name__}] {ex}")
        else:
            single_target.imageDropped.emit(img, seqlen, path, "")

    def _select_path(path: str) -> None:
        nonlocal open_default, open_path, save_path
        open_default = path
        open_path = path
        save_path = f"{os.path.splitext(path)[0]}.txt"

        save_action.setVisible(True)
        save_action.setToolTip(f"[%s] Save tags to: {save_path}")
        saveas_action.setModifiers(Qt.KeyboardModifier.ShiftModifier)

        prev_btn.setVisible(True)
        next_btn.setVisible(True)

        open_lbl.setText(os.path.basename(path))
        open_lbl.setToolTip(path)

        if not path_txt.text():
            path_txt.setText(os.path.dirname(path))

    def _select_other(url: str) -> None:
        nonlocal open_path, save_path
        open_path = ""
        save_path = ""

        prev_btn.setVisible(False)
        next_btn.setVisible(False)

        save_action.setVisible(False)
        saveas_action.setModifiers(None)

        if url:
            open_lbl.setText("🌐 " + url.rsplit("/", maxsplit=1)[1])
            open_lbl.setToolTip(url)
        else:
            open_lbl.setText("📋")
            open_lbl.setToolTip("")

    def _select_failed_path(path: str) -> None:
        nonlocal single_input, single_features, single_output
        assert model is not None

        single_input = None
        single_features = None
        single_output = None

        _select_path(path)

        single_image.clear()
        single_image.setText("<Failed to Load>")

        rerun_btn.setEnabled(False)
        analysis_btn.setEnabled(False)
        analysis_btn.setChecked(False)

        single_txt.clearItems()

        try:
            _rewriter((
                aliases_value(),
                prefix_by_ck.isChecked(),
                spaces_ck.isChecked(),
                escape_ck.isChecked(),
            ))
        except Exception as ex:
            QMessageBox.warning(root, "Configuration Issue", f"[{ex.__class__.__name__}] {ex}")

        if rewriter is not None:
            single_txt.setItems(
                ((label, 0.0) for label in model.labels),
                {}, rewriter
            )
            _enableEditActions(True)
        else:
            _enableEditActions(False)

        try:
            with open(save_path, "r", encoding="utf8") as file:
                tag_str = file.read()
        except:
            pass
        else:
            single_txt.setPlainText(tag_str)

    def _reset_single_image() -> None:
        assert single_input is not None

        single_image.clear()
        single_image.setPixmap(QPixmap.fromImage(QImage(
            single_input.numpy().data,
            single_input.size(1),
            single_input.size(0),
            single_input.size(1) * 3,
            QImage.Format.Format_RGB888
        )))

        analysis_btn.setChecked(False)

    @connect(single_target.imageDropped)
    def _image_dropped(img: Image, seqlen: int, path: str, url: str) -> None:
        nonlocal single_seqlen, single_input, single_features, single_output
        nonlocal single_preview, single_preserved
        single_seqlen = seqlen
        single_input = image.as_tensor(img)
        single_features = None
        single_output = None
        single_preview = False
        single_preserved = ""

        if path:
            _select_path(path)

            try:
                with open(save_path, "r", encoding="utf8") as file:
                    single_preserved = file.read()
            except:
                pass
            else:
                single_preview = Qt.KeyboardModifier.ShiftModifier in QGuiApplication.queryKeyboardModifiers()
        else:
            _select_other(url)

        _reset_single_image()
        _run_single()

    @connect(rerun_action.triggered)
    def _rerun_action() -> None:
        nonlocal single_preview, single_preserved

        single_preview = False
        single_preserved = single_txt.toPlainText()
        _run_single()

    @connect(preview_action.triggered)
    def _preview_action() -> None:
        nonlocal single_preview, single_preserved

        single_preview = True
        single_preserved = single_txt.toPlainText()
        _run_single()

    cam_idx = 0

    @connect(cam_action.triggered)
    def _cam_action() -> None:
        nonlocal cam_idx
        assert model is not None

        if not analysis_btn.isChecked():
            _reset_single_image()
            return

        labels = sorted(model.label_names())

        cam_tag, ok = QInputDialog.getItem(
            root, "Attention Visualization", "Select a tag to visualize...",
            labels, current=cam_idx, editable=False
        )
        if not ok:
            analysis_btn.setChecked(False)
            return

        cam_idx = labels.index(cam_tag)
        _run_cam(cam_tag)

    @connect(pca_action.triggered)
    def _pca_action() -> None:
        if not analysis_btn.isChecked():
            _reset_single_image()
            return

        dialog = QMessageBox(root)
        dialog.setWindowTitle("PCA")
        dialog.setText("Run Principal Component Analysis on...")

        depth_0_btn = dialog.addButton("Features", QMessageBox.ButtonRole.AcceptRole)
        depth_1_btn = dialog.addButton("Outputs", QMessageBox.ButtonRole.AcceptRole)
        cancel_btn = dialog.addButton(QMessageBox.StandardButton.Cancel)

        dialog.setDefaultButton(depth_0_btn)
        dialog.exec()

        if dialog.clickedButton() is depth_0_btn:
            _run_pca(0)
        elif dialog.clickedButton() is depth_1_btn:
            _run_pca(1)
        else:
            analysis_btn.setChecked(False)

    @connect(clean_action.triggered)
    def _clean_action() -> None:
        single_txt.cleanupItems()

    @connect(sort_action.triggered)
    def _sort_action() -> None:
        single_txt.cleanupItems(sort=True)

    @connect(copy_btn.clicked)
    def _copy_btn() -> None:
        text = single_txt.toPlainText().strip()
        if text:
            QGuiApplication.clipboard().setText(text)

    @connect(save_action.triggered)
    def _save_action() -> None:
        try:
            with open(save_path, "w", encoding="utf-8") as file:
                file.write(single_txt.toPlainText().strip())
        except Exception as ex:
            QMessageBox.warning(root, "Failed to Save", f"[{ex.__class__.__name__}] {ex}")

    @connect(saveas_action.triggered)
    def _saveas_action() -> None:
        nonlocal save_path

        path, _ = QFileDialog.getSaveFileName(
            root, f"Save Tags", save_path, "Text (*.txt);;Any (*)"
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as file:
                file.write(single_txt.toPlainText().strip())
        except Exception as ex:
            QMessageBox.warning(root, "Failed to Save", f"[{ex.__class__.__name__}] {ex}")
            return

        save_path = path
        save_action.setVisible(True)
        save_action.setToolTip(f"[%s] Save tags to: {save_path}")
        saveas_action.setModifiers(Qt.KeyboardModifier.ShiftModifier)

    @connect(ensure_implied_btn.clicked)
    def _ensure_implied_btn() -> None:
        single_txt.cleanupItems(ensure_implied=True)

    @connect(csv_copy_action.triggered)
    def _csv_copy_action() -> None:
        with StringIO(newline="") as io:
            _write_csv(io)
            csv = io.getvalue()

        mime = QMimeData()
        mime.setData("text/csv;charset=UTF-8", csv.encode())
        mime.setText(csv)
        QGuiApplication.clipboard().setMimeData(mime)

    @connect(csv_save_action.triggered)
    def _csv_save_action() -> None:
        if save_path.endswith(".txt"):
            path = f"{save_path[:-4]}.csv"
        elif save_path:
            path = os.path.join(os.path.dirname(save_path), "tags.csv")
        else:
            path = "tags.csv"

        path, _ = QFileDialog.getSaveFileName(
            root, "Save CSV", path, "CSV (*.csv);;Any (*)"
        )
        if not path:
           return

        try:
            with open(path, "w", encoding="utf-8") as file:
                _write_csv(file)
        except Exception as ex:
            QMessageBox.warning(root, "Failed to Save", f"[{ex.__class__.__name__}] {ex}")
            return

    def _write_csv(io) -> None:
        writer = csv.writer(io)
        writer.writerow(("tag", "category", "probability"))
        for text, label, prob, checked in single_txt.getItems():
            if checked:
                writer.writerow((
                    text,
                    label.category if label is not None else "",
                    f"{prob:.4}",
                ))

    @connect(single_txt.addExclusion)
    def _add_exclusion(exclusion: str) -> None:
        if excl_tags_txt.isReadOnly():
            return

        text = excl_tags_txt.toPlainText()
        exclusions = parse_list(text)

        if exclusion in exclusions:
            return

        if exclusion.startswith("~~"):
            if exclusion[1:] in exclusions:
                return
        elif not exclusion.startswith("~"):
            if f"~{exclusion}" in exclusions:
                return

        if (last := text.rpartition("\n")[2]):
            if "#" in last:
                text += "\n"
            elif not last.endswith((" ", "\t")):
                text += " "

        text += exclusion

        excl_tags_txt.setPlainText(text)

    @connect(single_txt.visualizeTag)
    def _visualize_tag(label: str) -> None:
        nonlocal cam_idx
        assert model is not None

        if (
            not main_tabs.isTabEnabled(1)
            or not analysis_btn.isEnabled()
            or single_input is None
            or single_features is None
        ):
            return

        cam_idx = sorted(model.label_names()).index(label)

        main_tabs.setCurrentIndex(1)
        analysis_btn.setChecked(True)

        _run_cam(label)

    @connect(single_txt.showTagInfo)
    def _show_tag_info(label: str):
        config_tabs.setCurrentIndex(3)

        tr_item = tag_info_items[label]
        info_tags_tr.expandItem(tr_item)
        info_tags_tr.scrollToItem(tr_item, QAbstractItemView.ScrollHint.PositionAtTop)
        info_tags_tr.setFocus(Qt.FocusReason.OtherFocusReason)
        info_tags_tr.setCurrentItem(tr_item)

    @connect(config_tabs.currentChanged)
    def _config_tab(index: int) -> None:
        nonlocal stale_calibration
        if index != 3:
            stale_calibration = False
            return

        new_calibration_cfg = (
            metric_cb.currentText(),
            metric_arg_sb.value(),
            min_prec_sb.value()
        )
        if new_calibration_cfg == calibration_cfg:
            stale_calibration = False
            info_status_lbl.setText("")
            return

        if worker is not None:
            stale_calibration = True
            info_status_lbl.setText("(Calibration Changed)")
            return

        _sync_calibration()

    def _sync_calibration() -> None:
        nonlocal stale_calibration
        stale_calibration = False

        info_status_lbl.setText("Calibrating...")
        info_header.activate()
        config_tabs.repaint()

        _calibrate((
            metric_cb.currentText(),
            metric_arg_sb.value(),
            min_prec_sb.value()
        ))
        info_status_lbl.setText("")

    def _run_single() -> None:
        nonlocal worker
        assert worker is None
        assert single_input is not None

        main_tabs.setTabEnabled(0, False)
        single_image.setAcceptDrops(False)
        prev_btn.setEnabled(False)
        next_btn.setEnabled(False)
        open_btn.setEnabled(False)
        rerun_btn.setEnabled(False)
        analysis_btn.setEnabled(False)

        single_txt.clearItems()
        single_txt.setPlainText(single_preserved)
        single_txt.setReadOnly(True)
        _enableEditActions(False)

        prefixes: dict[str, str] = {}
        if prefix_by_ck.isChecked():
            prefixes["artist"] = "by_"

        worker = Thread(target=_single_worker, kwargs={
            "new_rewriter_cfg": (
                aliases_value(),
                prefix_by_ck.isChecked(),
                spaces_ck.isChecked(),
                escape_ck.isChecked(),
            ),
            "new_calibration_cfg": (metric_cb.currentText(), metric_arg_sb.value(), min_prec_sb.value()),
            "varlen_attn": varlen_ck.isChecked(),
            "implications": implications_cb.currentText(),
            "exclude_categories": {
                cat
                for cat, ck in excl_cat_cks.items()
                if not ck.isChecked()
            },
            "exclude_tags": excl_tags_value(),
            "exclusive_groups": excl_groups_value(),
            "prefix": prefix_txt.text(),
            "shuffle": shuffle_ck.isChecked(),
        }, name="single-worker", daemon=True)
        worker.start()

    @torch.no_grad()
    def _single_worker(
        *,
        new_rewriter_cfg: tuple[str, bool, bool, bool],
        new_calibration_cfg: tuple[str, float, float],
        varlen_attn: bool,
        implications: str,
        exclude_categories: set[str],
        exclude_tags: str,
        exclusive_groups: str,
        prefix: str,
        shuffle: bool,
    ) -> None:
        nonlocal single_features, single_output
        assert model is not None
        assert single_input is not None

        try:
            excl_tags = set(parse_list(exclude_tags))
            excl_groups = list(ExclusiveGroup.parse_file(exclusive_groups))

            _rewriter(new_rewriter_cfg)
            _calibrate(new_calibration_cfg)

            assert rewriter is not None
            assert calibration is not None

            if single_output is None:
                if varlen_attn:
                    patches, sizes, cu_seq = image.varlen([single_input], 16, device=device)
                    single_features = model.forward_features_varlen(model.from_srgb(patches), sizes, cu_seq, single_seqlen)
                    single_output = model.forward_head_varlen(**single_features)[0] # type: ignore
                else:
                    patches, sizes = image.stack([single_input], 16, single_seqlen, device=device)
                    single_features = model.forward_features(model.from_srgb(patches), sizes)
                    single_output = model.forward_head(**single_features)[0] # type: ignore

                single_output = single_output.cpu()

            labels = calibration.classify_output(
                single_output,
                implications=implications,
                exclude_categories=exclude_categories,
                exclude_labels=excl_tags,
                exclusive_groups=excl_groups,
                sort=True,
            )

            tag_str = rewriter.rewrite_join(
                labels.keys(),
                prefix=prefix,
                shuffle=shuffle,
            )

            proxy.invoke(_single_worker_done, (labels, tag_str))
        except Exception as ex:
            single_features = None
            single_output = None

            print_exc()
            proxy.invoke(_single_worker_done, f"[{ex.__class__.__name__}] {ex}")

    def _single_worker_done(result: str | tuple[dict[str, float], str]) -> None:
        nonlocal worker
        assert model is not None

        assert worker is not None
        worker.join()
        worker = None

        main_tabs.setTabEnabled(0, True)
        config_tabs.setTabEnabled(4, True)
        config_tabs.setCurrentIndex(4)

        single_target.setAcceptDrops(True)
        prev_btn.setEnabled(True)
        next_btn.setEnabled(True)
        open_btn.setEnabled(True)
        rerun_btn.setEnabled(True)
        single_txt.setReadOnly(False)

        if isinstance(result, str):
            analysis_btn.setEnabled(False)

            if rewriter is not None:
                single_txt.setItems(
                    ((label, 0.0) for label in model.labels),
                    {}, rewriter
                )
                _enableEditActions(True)
            else:
                _enableEditActions(False)

            single_txt.setPlainText(single_preserved)

            QMessageBox.warning(root, "Failed to Classify", result)
            return

        assert rewriter is not None
        assert single_output is not None

        analysis_btn.setEnabled(True)

        single_txt.setItems(
            zip(model.labels, single_output.tolist()),
            result[0], rewriter,
        )
        _enableEditActions(True)

        if single_preview:
            single_txt.setPlainText(single_preserved)
        else:
            single_txt.setPlainText(result[1])

    def _run_cam(cam_tag: str) -> None:
        nonlocal worker
        assert worker is None

        main_tabs.setTabEnabled(0, False)

        single_image.setAcceptDrops(False)
        prev_btn.setEnabled(False)
        next_btn.setEnabled(False)
        open_btn.setEnabled(False)
        rerun_btn.setEnabled(False)
        analysis_btn.setEnabled(False)

        worker = Thread(target=_cam_worker, args=(cam_tag,), name="cam-worker", daemon=True)
        worker.start()

    @torch.no_grad()
    def _cam_worker(cam_tag: str) -> None:
        assert model is not None
        assert single_input is not None
        assert single_features is not None

        patch_h = single_input.size(0) // 16
        patch_w = single_input.size(1) // 16

        try:
            model.select_labels(cam_tag)
            model.head.logit = True

            features = single_features["features"]
            features.requires_grad_(True).retain_grad()

            with torch.enable_grad():
                if "cu_seq" in single_features:
                    model.forward_head_varlen(**single_features)[0, 0].backward()
                else:
                    model.forward_head(**single_features)[0, 0].backward()

            cam = features.grad
            features.grad = None

            assert cam is not None
            cam = cam[0, :patch_h*patch_w]
            features = features[0, :patch_h*patch_w]

            cam = torch.linalg.vecdot(cam, features.sign())
            cam = rearrange(cam, "(h w) -> h 1 w 1", h=patch_h, w=patch_w)

            cam_a = cam.abs().float()
            cam_a.mul_(1.0 / cam_a.amax())
            cam_a = image.srgb_to_rgb(cam_a)
            cam_a.mul_(0.5)

            cam_r = torch.where(cam < 0, cam_a, 0.0)
            cam_g = torch.where(cam > 0, cam_a, 0.0)
            cam_a = 1.0 - cam_a
            del cam

            cam_c = single_input.to(device=device, dtype=torch.float, non_blocking=True, copy=True)
            cam_c = rearrange(cam_c, "(h p1) (w p2) c -> h p1 w p2 c", p1=16, p2=16)
            cam_c.mul_(1/255)
            cam_c = image.srgb_to_rgb(cam_c)

            cam_c.lerp_(image.rgb_to_grey(cam_c, keepdim=True), (2/3))
            cam_c.mul_(cam_a.unsqueeze(-1))
            cam_c[..., 0].add_(cam_r)
            cam_c[..., 1].add_(cam_g)
            del cam_a, cam_r, cam_g

            cam_c = image.rgb_to_srgb(cam_c)
            cam_c.mul_(255.0).round_()
            cam_c = rearrange(cam_c, "h p1 w p2 c -> (h p1) (w p2) c")
            cam_c = cam_c.to(
                device="cpu", dtype=torch.uint8,
                memory_format=torch.contiguous_format
            )
            proxy.invoke(_cam_worker_done, cam_c)
        except Exception as ex:
            print_exc()
            proxy.invoke(_cam_worker_done, f"[{ex.__class__.__name__}] {ex}")
        finally:
            model.select_labels(None)
            model.head.logit = False # type: ignore

    def _cam_worker_done(result: str | Tensor) -> None:
        nonlocal worker

        assert worker is not None
        worker.join()
        worker = None

        main_tabs.setTabEnabled(0, True)
        single_target.setAcceptDrops(True)
        prev_btn.setEnabled(True)
        next_btn.setEnabled(True)
        open_btn.setEnabled(True)
        rerun_btn.setEnabled(True)
        analysis_btn.setEnabled(True)

        if isinstance(result, str):
            analysis_btn.setChecked(False)
            QMessageBox.warning(root, "Failed to Run CAM", result)
            return

        single_image.setPixmap(QPixmap.fromImage(QImage(
            result.numpy().data,
            result.size(1),
            result.size(0),
            result.size(1) * 3,
            QImage.Format.Format_RGB888
        )))

    def _run_pca(depth: int) -> None:
        nonlocal worker
        assert worker is None

        main_tabs.setTabEnabled(0, False)
        single_image.setAcceptDrops(False)
        prev_btn.setEnabled(False)
        next_btn.setEnabled(False)
        open_btn.setEnabled(False)
        rerun_btn.setEnabled(False)
        analysis_btn.setEnabled(False)

        worker = Thread(target=_pca_worker, args=(depth,), name="pca-worker", daemon=True)
        worker.start()

    @torch.no_grad()
    def _pca_worker(depth: int) -> None:
        assert model is not None
        assert single_input is not None
        assert single_features is not None

        patch_h = single_input.size(0) // 16
        patch_w = single_input.size(1) // 16

        try:
            if depth == 0:
                features = single_features["features"]
            elif "cu_seq" in single_features:
                features = model.attn_pool.forward_unpooled_varlen(**single_features)
            else:
                features = model.attn_pool.forward_unpooled(
                    single_features["features"],
                    model._attn_mask(single_features["valid"]), # type: ignore[typeddict-item]
                )

            features = features[0, :patch_h*patch_w].float()

            torch.manual_seed(0)
            v = torch.pca_lowrank(features, niter=4, q=256)[2]

            pca = features @ v[:, [1, 0, 2]] # GRB
            pca.sub_(pca.amin(dim=0))
            pca.mul_(1.0 / pca.amax(dim=0))
            pca = image.srgb_to_rgb(pca)

            pca = rearrange(pca, "(h w) c -> h 1 w 1 c", h=patch_h, w=patch_w)

            pca_c = single_input.to(device=device, dtype=torch.float, non_blocking=True, copy=True)
            pca_c = rearrange(pca_c, "(h p1) (w p2) c -> h p1 w p2 c", p1=16, p2=16)
            pca_c.mul_(1/255)
            pca_c = image.srgb_to_rgb(pca_c)

            pca_c.mul_(0.75).lerp_(pca, 0.5)
            del pca

            pca_c = image.rgb_to_srgb(pca_c)
            pca_c.mul_(255.0).round_()
            pca_c = rearrange(pca_c, "h p1 w p2 c -> (h p1) (w p2) c")
            pca_c = pca_c.to(
                device="cpu", dtype=torch.uint8,
                memory_format=torch.contiguous_format
            )
            proxy.invoke(_pca_worker_done, pca_c)
        except Exception as ex:
            print_exc()
            proxy.invoke(_pca_worker_done, f"[{ex.__class__.__name__}] {ex}")

    def _pca_worker_done(result: str | Tensor) -> None:
        nonlocal worker

        assert worker is not None
        worker.join()
        worker = None

        main_tabs.setTabEnabled(0, True)
        single_target.setAcceptDrops(True)
        prev_btn.setEnabled(True)
        next_btn.setEnabled(True)
        open_btn.setEnabled(True)
        rerun_btn.setEnabled(True)
        analysis_btn.setEnabled(True)

        if isinstance(result, str):
            analysis_btn.setChecked(False)
            QMessageBox.warning(root, "Failed to Run PCA", result)
            return

        single_image.setPixmap(QPixmap.fromImage(QImage(
            result.numpy().data,
            result.size(1),
            result.size(0),
            result.size(1) * 3,
            QImage.Format.Format_RGB888
        )))

    @connect(about_qt_btn.clicked)
    def _about_qt_btn() -> None:
        QMessageBox.aboutQt(root, "About Qt")

    root.show()

    Thread(target=_startup, daemon=True).start()

    try:
        with open("data/Aliases-2024-04-07.txt") as aliases_file:
            aliases_txt.setPlainText(aliases_file.read())
    except:
        pass

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
