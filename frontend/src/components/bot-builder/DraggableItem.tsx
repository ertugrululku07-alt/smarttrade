import React from 'react';

interface DraggableItemProps {
    item: {
        id: string;
        type: string;
        title: string;
        desc: string;
    };
}

export function DraggableItem({ item }: DraggableItemProps) {
    const handleDragStart = (e: React.DragEvent) => {
        e.dataTransfer.setData('application/json', JSON.stringify({
            id: item.id,
            type: item.type,
            title: item.title
        }));
        e.dataTransfer.effectAllowed = 'copy';
    };

    let bgColor = 'bg-slate-700';
    let borderColor = 'border-slate-600';
    let textColor = 'text-slate-200';

    if (item.type === 'indicator') {
        bgColor = 'bg-blue-900/40 hover:bg-blue-800/60';
        borderColor = 'border-blue-700/50';
        textColor = 'text-blue-200';
    } else if (item.type === 'logic') {
        bgColor = 'bg-purple-900/40 hover:bg-purple-800/60';
        borderColor = 'border-purple-700/50';
        textColor = 'text-purple-200';
    } else if (item.type === 'action') {
        bgColor = 'bg-emerald-900/40 hover:bg-emerald-800/60';
        borderColor = 'border-emerald-700/50';
        textColor = 'text-emerald-200';
    }

    return (
        <div
            draggable
            onDragStart={handleDragStart}
            className={`p-4 rounded-xl border ${bgColor} ${borderColor} cursor-grab active:cursor-grabbing transition-colors shadow-sm`}
        >
            <h4 className={`font-semibold ${textColor} mb-1`}>{item.title}</h4>
            <p className="text-xs text-slate-400">{item.desc}</p>
        </div>
    );
}
