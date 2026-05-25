// Copyright (C) 2025 Marco Hernandez <ragettyandy@gmail.com>
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// For information contact Marco Hernandez <ragettyandy@gmail.com>

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
