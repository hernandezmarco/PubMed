async function deleteCollection(id) {
  if (!confirm('Delete this collection?')) return;
  Progress.start(8, 75);
  try {
    const res = await fetch(`/collections/${id}/delete`, { method: 'POST' });
    if ((await res.json()).ok) {
      const card = document.getElementById(`card-${id}`);
      card.classList.add('removing');
      setTimeout(() => card.remove(), 260);
      Progress.done();
    } else {
      Progress.error();
    }
  } catch {
    Progress.error();
  }
}
