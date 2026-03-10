import React from 'react';

interface SortableBlockProps {
    id: string;
    title: string;
    type: string;
}

export function SortableBlock({ id, title, type }: SortableBlockProps) {
    let headerColor = 'bg-slate-700';
    let badgeColor = 'bg-slate-600 text-slate-200';

    if (type === 'indicator') {
        headerColor = 'bg-blue-600';
        badgeColor = 'bg-blue-500/20 text-blue-300 border-blue-500/30';
    } else if (type === 'logic') {
        headerColor = 'bg-purple-600';
        badgeColor = 'bg-purple-500/20 text-purple-300 border-purple-500/30';
    } else if (type === 'action') {
        headerColor = 'bg-emerald-600';
        badgeColor = 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30';
    }

    return (
        <div className="relative w-full bg-slate-800 rounded-xl border border-slate-700 shadow-xl overflow-hidden group">
            {/* Node Header */}
            <div className={`px-4 py-3 ${headerColor} flex justify-between items-center`}>
                <div className="flex items-center gap-3">
                    {/* Icon */}
                    <svg className="w-5 h-5 text-white/50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8h16M4 16h16" />
                    </svg>
                    <h3 className="font-bold text-white tracking-wide">{title}</h3>
                </div>
                <span className={`text-[10px] font-bold uppercase px-2.5 py-1 rounded-full border ${badgeColor}`}>
                    {type}
                </span>
            </div>

            {/* Node Body / Inputs */}
            <div className="p-4 space-y-4">
                {type === 'indicator' && (
                    <div className="flex gap-4">
                        <div className="flex-1">
                            <label className="block text-xs font-medium text-slate-400 mb-1">Period</label>
                            <input type="number" defaultValue={14} className="w-full bg-slate-900 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500 transition-colors" />
                        </div>
                        <div className="flex-1">
                            <label className="block text-xs font-medium text-slate-400 mb-1">Source</label>
                            <select className="w-full bg-slate-900 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:border-blue-500 transition-colors">
                                <option>Close</option>
                                <option>Open</option>
                                <option>High</option>
                                <option>Low</option>
                            </select>
                        </div>
                    </div>
                )}

                {type === 'logic' && (
                    <div className="flex items-center gap-3">
                        <span className="text-sm font-medium text-slate-300">Previous Node</span>
                        <select className="bg-slate-900 border border-slate-700 rounded-md px-2 py-1 text-sm text-white focus:outline-none focus:border-purple-500">
                            <option>&gt;</option>
                            <option>&lt;</option>
                            <option>=</option>
                            <option>&gt;=</option>
                            <option>&lt;=</option>
                        </select>
                        <input type="number" defaultValue={30} className="w-24 bg-slate-900 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:border-purple-500 transition-colors" />
                    </div>
                )}

                {type === 'action' && (
                    <div className="flex items-center gap-4">
                        <div className="flex-1">
                            <label className="block text-xs font-medium text-slate-400 mb-1">Amount (%)</label>
                            <input type="number" defaultValue={100} className="w-full bg-slate-900 border border-slate-700 rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:border-emerald-500 transition-colors" />
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
