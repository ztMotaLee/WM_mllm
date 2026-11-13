"""Persist provider attribution and usage metadata without request payloads."""
from __future__ import annotations

from pathlib import Path
import uuid

from ..logging_utils import sanitize
from ..providers.factory import Providers
from .io import IntegrityError, resolve_run_path, write_once


class ReceiptClient:
    """Transparent injected-client boundary; requests and results pass unchanged."""

    def __init__(self, client, *, kind, run_root, secrets):
        self.client = client
        self.route = client.route
        self.kind = kind
        self.run_root = Path(run_root).resolve()
        self.secrets = tuple(secrets)

    def _record(self, receipt):
        if receipt is None:
            return
        value = sanitize(receipt.to_record(), secrets=self.secrets)
        path = self.run_root / 'receipts' / (self.kind + '-' + uuid.uuid4().hex + '.json')
        write_once(path, value, run_root=self.run_root)

    def _invoke(self, method, request):
        if method == 'edit':
            try:
                relative = Path(request.output_dir).relative_to(self.run_root).as_posix()
            except ValueError:
                raise IntegrityError('Image output directory lies outside the run root') from None
            resolve_run_path(self.run_root, relative, must_exist=False)
        try:
            result = getattr(self.client, method)(request)
        except Exception as error:
            self._record(getattr(error, 'receipt', None))
            raise
        self._record(result.receipt)
        return result

    def complete(self, request):
        return self._invoke('complete', request)

    def edit(self, request):
        return self._invoke('edit', request)

    def close(self):
        self.client.close()


def recorded_providers(providers, *, run_root, secrets):
    def wrap(client, kind):
        if isinstance(client, ReceiptClient):
            if client.run_root != Path(run_root).resolve():
                raise IntegrityError('Provider receipts belong to another run root')
            return client
        return ReceiptClient(client, kind=kind, run_root=run_root, secrets=secrets)
    return Providers(wrap(providers.chat, 'chat'), wrap(providers.image, 'image'), providers.video)
